#!/usr/bin/env python3
"""
HTML Report Generator - Generate HTML reports with sortable tables for covered calls analysis.

This module handles the generation of HTML output with embedded CSS and JavaScript
for displaying and sorting tabular data.

NOTE: This file has been refactored. The code now uses modular components from
scripts.html_report package. For new code, import directly from scripts.html_report.
"""

import pandas as pd
import numpy as np
import sys
import textwrap
import html
import re
from pathlib import Path
from datetime import datetime

# Import from new modular structure
try:
    from scripts.html_report import generate_html_output as _generate_html_output
    from scripts.html_report.constants import COMPACT_HEADER_MAP
    from scripts.html_report.formatters import (
        format_age_seconds,
        format_numeric_value,
        truncate_header,
        normalize_col_name as _normalize_col_name_helper
    )
    from scripts.html_report.css_generator import get_css_styles
    from scripts.html_report.js_generator import get_javascript
    _USE_NEW_MODULES = True
except ImportError:
    # Fallback to local definitions if modules don't exist yet
    _USE_NEW_MODULES = False
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


def format_age_seconds(age_seconds):
    """Format age in seconds to human-readable format.
    
    Args:
        age_seconds: Age in seconds (float or numeric string)
        
    Returns:
        Human-readable string like "5 secs", "2 mins", "3 hrs", "1.2 days"
    """
    if pd.isna(age_seconds) or age_seconds == '' or age_seconds is None:
        return ''
    
    try:
        # Convert to float
        if isinstance(age_seconds, str):
            # Try to extract first valid number from string
            match = re.search(r'-?\d+\.?\d*', age_seconds)
            if match:
                age_sec = float(match.group())
            else:
                return str(age_seconds)
        else:
            age_sec = float(age_seconds)
        
        # Handle negative or zero values
        if age_sec < 0:
            return 'N/A'
        if age_sec == 0:
            return '0 secs'
        
        # Convert to appropriate unit
        if age_sec < 60:
            # Less than 1 minute - show seconds
            return f"{age_sec:.1f} secs" if age_sec < 10 else f"{int(age_sec)} secs"
        elif age_sec < 3600:
            # Less than 1 hour - show minutes
            mins = age_sec / 60
            return f"{mins:.1f} mins" if mins < 10 else f"{int(mins)} mins"
        elif age_sec < 86400:
            # Less than 1 day - show hours
            hrs = age_sec / 3600
            return f"{hrs:.1f} hrs" if hrs < 10 else f"{int(hrs)} hrs"
        else:
            # 1 day or more - show days
            days = age_sec / 86400
            return f"{days:.1f} days" if days < 10 else f"{int(days)} days"
    except (ValueError, TypeError, AttributeError):
        return str(age_seconds) if age_seconds != '' else ''


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
        # Handle malformed strings like '0.120.260.110.210.36'
        val_str = str(x)
        # Try direct conversion first
        try:
            val = float(val_str)
        except (ValueError, TypeError):
            # Extract first valid number from string if direct conversion fails
            match = re.search(r'-?\d+\.?\d*', val_str)
            if match:
                val = float(match.group())
            else:
                # If no number found, return the string as-is
                return str(x) if x != '' else ''
        
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
    except (ValueError, TypeError, AttributeError):
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
            display: block !important;
        }
        
        .table-wrapper {
            overflow-x: auto;
            padding: 20px;
            display: block;
        }
        
        /* Hidden columns handling */
        .table-wrapper.hide-hidden th.is-hidden-col,
        .table-wrapper.hide-hidden td.is-hidden-col {
            display: none;
        }
        
        /* Always hidden columns (hidden in all cases) */
        th.always-hidden,
        td.always-hidden {
            display: none !important;
        }
        
        /* Hide group header row when hidden columns are shown */
        .table-wrapper:not(.hide-hidden) tr.group-header-row {
            display: none;
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9em;
            table-layout: auto;
        }
        
        thead {
            background: #f8f9fa;
            position: sticky;
            top: 0;
            z-index: 10;
        }
        
        th {
            padding: 15px 8px;
            text-align: center;
            font-weight: 600;
            color: #333;
            border-bottom: 2px solid #dee2e6;
            cursor: pointer;
            user-select: none;
            white-space: normal;
            word-wrap: break-word;
            word-break: break-word;
            width: auto;
            max-width: 12ch;
            min-width: fit-content;
            line-height: 1.3;
            position: relative;
        }
        
        /* Group header row styling */
        tr.group-header-row th.group-header {
            background: #667eea;
            color: white;
            text-align: center;
            font-weight: 700;
            font-size: 0.95em;
            border-bottom: 2px solid #5568d3;
            cursor: default;
            padding: 10px 12px;
        }
        
        tr.group-header-row th.group-header.is-hidden-col {
            display: none;
        }
        
        tr.group-header-row th.group-header:hover {
            background: #5568d3;
        }
        
        /* Column header row styling (second row) */
        tr.column-header-row th {
            border-top: 1px solid #dee2e6;
            padding: 10px 12px;
            font-size: 0.9em;
        }
        
        th.even-col {
            background: #f0f0f0;
        }
        
        th:hover {
            background: #e9ecef;
        }
        
        th.even-col:hover {
            background: #d0d0d0;
        }
        
        
        .column-name-display {
            display: block;
        }
        
        .column-name-short-long {
            display: none;
            font-weight: 600;
            font-size: 0.85em;
            color: #495057;
        }
        
        .column-name-filterable {
            display: none;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
            color: #667eea;
        }
        
        /* When filter section is NOT expanded, show short/long labels for grouped columns */
        th.grouped-column:not(.showing-filterable) .column-name-display {
            display: none;
        }
        
        th.grouped-column:not(.showing-filterable) .column-name-short-long {
            display: block;
        }
        
        th.showing-filterable .column-name-display {
            display: none;
        }
        
        th.showing-filterable .column-name-short-long {
            display: none;
        }
        
        th.showing-filterable .column-name-filterable {
            display: block;
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
            padding: 12px 8px;
            border-bottom: 1px solid #dee2e6;
            color: #495057;
            text-align: center;
            white-space: normal;
            word-wrap: break-word;
            word-break: break-word;
            width: auto;
            max-width: 12ch;
            min-width: fit-content;
        }
        
        /* Wrap text if content exceeds 12 characters */
        td, th {
            overflow-wrap: break-word;
            word-break: break-word;
        }
        
        td.even-col {
            background: #f9f9f9;
        }
        
        td.price-positive {
            color: #28a745;
            font-weight: 600;
        }
        
        td.price-negative {
            color: #dc3545;
            font-weight: 600;
        }
        
        tbody tr.even-row {
            background: #f8f9fa;
        }
        
        tbody tr.even-row td.even-col {
            background: #f0f0f0;
        }
        
        tbody tr:hover {
            background: #e9ecef !important;
        }
        
        tbody tr:hover td {
            background: #e9ecef !important;
        }
        
        tbody tr.even-row:hover {
            background: #d9d9d9 !important;
        }
        
        tbody tr.even-row:hover td {
            background: #d9d9d9 !important;
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
        
        .filter-section {
            background: #f8f9fa;
            padding: 20px;
            border-bottom: 2px solid #dee2e6;
            margin-bottom: 20px;
            display: none; /* Hidden by default */
        }
        
        .filter-section.expanded {
            display: block;
        }
        
        .filter-controls {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            align-items: center;
            margin-bottom: 15px;
        }
        
        .filter-input-group {
            display: flex;
            align-items: center;
            gap: 5px;
            flex: 1;
            min-width: 250px;
        }
        
        .filter-input {
            flex: 1;
            padding: 10px;
            border: 2px solid #dee2e6;
            border-radius: 6px;
            font-size: 0.9em;
            font-family: 'Courier New', monospace;
        }
        
        .filter-input:focus {
            outline: none;
            border-color: #667eea;
        }
        
        .filter-button {
            padding: 10px 20px;
            background: #667eea;
            color: white;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-size: 0.9em;
            font-weight: 600;
            transition: background 0.3s ease;
        }
        
        .filter-button:hover {
            background: #5568d3;
        }
        
        .filter-button.clear {
            background: #6c757d;
        }
        
        .filter-button.clear:hover {
            background: #5a6268;
        }
        
        .filter-logic {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 10px;
        }
        
        .filter-logic label {
            font-weight: 600;
            color: #495057;
        }
        
        .filter-logic input[type="radio"] {
            margin-right: 5px;
        }
        
        .filter-help {
            font-size: 0.85em;
            color: #6c757d;
            margin-top: 10px;
            padding: 10px;
            background: white;
            border-radius: 5px;
            border-left: 3px solid #667eea;
        }
        
        .filter-help code {
            background: #f8f9fa;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
        }
        
        .filter-error {
            color: #dc3545;
            font-size: 0.9em;
            margin-top: 5px;
            padding: 5px;
            background: #f8d7da;
            border-radius: 3px;
        }
        
        /* Mobile Card Layout */
        .card-wrapper {
            display: none;
            padding: 15px;
        }
        
        .data-card {
            background: white;
            border: 1px solid #dee2e6;
            border-radius: 8px;
            margin-bottom: 15px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            overflow: hidden;
            transition: box-shadow 0.3s ease;
        }
        
        .data-card:hover {
            box-shadow: 0 4px 8px rgba(0,0,0,0.15);
        }
        
        .data-card.hidden {
            display: none;
        }
        
        .card-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 10px;
        }
        
        .card-header-main {
            flex: 1;
            min-width: 200px;
        }
        
        .card-ticker {
            font-size: 1.3em;
            font-weight: bold;
            margin-bottom: 5px;
        }
        
        .card-price {
            font-size: 1.1em;
            display: flex;
            align-items: center;
            gap: 10px;
            flex-wrap: wrap;
        }
        
        .card-price-value {
            font-weight: 600;
        }
        
        .card-price-change {
            font-size: 0.9em;
            padding: 2px 8px;
            border-radius: 4px;
            background: rgba(255,255,255,0.2);
        }
        
        .card-price-change.positive {
            background: rgba(40, 167, 69, 0.3);
        }
        
        .card-price-change.negative {
            background: rgba(220, 53, 69, 0.3);
        }
        
        .card-body {
            padding: 15px;
        }
        
        .card-row {
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid #f0f0f0;
            align-items: center;
        }
        
        .card-row:last-child {
            border-bottom: none;
        }
        
        .card-label {
            font-weight: 600;
            color: #495057;
            font-size: 0.9em;
            flex: 1;
        }
        
        .card-value {
            color: #212529;
            text-align: right;
            flex: 1;
            font-size: 0.95em;
        }
        
        .card-section {
            margin-bottom: 15px;
        }
        
        .card-section-title {
            font-weight: 700;
            color: #667eea;
            font-size: 0.95em;
            margin-bottom: 10px;
            padding-bottom: 5px;
            border-bottom: 2px solid #667eea;
        }
        
        .card-details {
            display: none;
        }
        
        .card-details.expanded {
            display: block;
        }
        
        .card-toggle {
            background: #f8f9fa;
            border: none;
            width: 100%;
            padding: 12px;
            cursor: pointer;
            font-size: 0.9em;
            color: #667eea;
            font-weight: 600;
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 8px;
            transition: background 0.3s ease;
        }
        
        .card-toggle:hover {
            background: #e9ecef;
        }
        
        .card-toggle-icon {
            transition: transform 0.3s ease;
        }
        
        .card-toggle.expanded .card-toggle-icon {
            transform: rotate(180deg);
        }
        
        .card-primary {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
            gap: 10px;
            margin-bottom: 10px;
        }
        
        .card-primary-item {
            background: #f8f9fa;
            padding: 10px;
            border-radius: 6px;
            text-align: center;
        }
        
        .card-primary-label {
            font-size: 0.75em;
            color: #6c757d;
            margin-bottom: 4px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .card-primary-value {
            font-size: 1.1em;
            font-weight: 600;
            color: #212529;
        }
        
        @media (max-width: 768px) {
            .header h1 {
                font-size: 1.8em;
            }
            
            /* Hide table on mobile */
            .table-wrapper {
                display: none !important;
            }
            
            /* Show cards on mobile */
            .card-wrapper {
                display: block !important;
            }
            
            table {
                font-size: 0.8em;
            }
            
            th, td {
                padding: 8px 6px;
            }
            
            .filter-controls {
                flex-direction: column;
            }
            
            .filter-input-group {
                width: 100%;
            }
            
            .container {
                max-width: 100%;
                border-radius: 0;
            }
            
            body {
                padding: 0;
            }
            
            .card-header {
                padding: 12px;
            }
            
            .card-ticker {
                font-size: 1.2em;
            }
            
            .card-body {
                padding: 12px;
            }
        }
        
        @media (min-width: 769px) {
            /* Hide cards on desktop */
            .card-wrapper {
                display: none !important;
            }
            
            /* Show table on desktop */
            .table-wrapper {
                display: block !important;
            }
            
            /* Show desktop-only messages */
            .desktop-only {
                display: block;
            }
            
            .mobile-only {
                display: none;
            }
        }
        
        .desktop-only {
            display: block;
        }
        
        .mobile-only {
            display: none;
        }
        
        /* Ensure table-wrapper is visible in active tabs on desktop */
        @media (min-width: 769px) {
            .tab-content.active .table-wrapper {
                display: block !important;
            }
            .tab-content.active .card-wrapper {
                display: none !important;
            }
        }
        
        @media (max-width: 768px) {
            .desktop-only {
                display: none;
            }
            
            .mobile-only {
                display: block;
            }
        }
"""


def get_javascript():
    """Get JavaScript code for table sorting and filtering functionality.
    
    Returns:
        String containing JavaScript code
    """
    return """        // Namespaced state for calls and puts
        let sortDirection = {};
        let currentSortColumn = {};
        let activeFilters = {};
        let filterLogic = {};
        
        // Initialize state for a prefix
        function initStateForPrefix(prefix) {
            if (!sortDirection[prefix]) sortDirection[prefix] = {};
            if (currentSortColumn[prefix] === undefined) currentSortColumn[prefix] = -1;
            if (!activeFilters[prefix]) activeFilters[prefix] = [];
            if (!filterLogic[prefix]) filterLogic[prefix] = 'AND';
        }
        
        // Column name mapping (original names to display names) - namespaced by table
        const columnMap = {};
        
        // Initialize column map for a specific table
        function initColumnMap(prefix) {
            const tableId = prefix + 'resultsTable';
            const table = document.getElementById(tableId);
            // Return early if table doesn't exist (prevents null reference errors)
            if (!table) {
                return;
            }
            // Only use column headers from the column-header-row, not group headers
            const headerRow = table.querySelector('tr.column-header-row');
            const headers = headerRow ? headerRow.querySelectorAll('th') : table.querySelectorAll('th');
            headers.forEach((th, index) => {
                const colName = th.textContent.trim().replace(/\\s+/g, ' ').replace(/\\n/g, '');
                columnMap[colName] = index;
                // Also map common variations
                const lowerName = colName.toLowerCase();
                if (lowerName.includes('pe') && lowerName.includes('ratio')) columnMap['pe_ratio'] = index;
                if (lowerName.includes('market') && lowerName.includes('cap')) columnMap['market_cap_b'] = index;
                if (lowerName.includes('current') || lowerName.includes('curr')) columnMap['current_price'] = index;
                if (lowerName.includes('strike')) columnMap['strike_price'] = index;
                if (lowerName.includes('premium') && !lowerName.includes('daily') && !lowerName.includes('total')) columnMap['option_premium'] = index;
                if (lowerName.includes('daily') && lowerName.includes('prem')) columnMap['daily_premium'] = index;
                if (lowerName.includes('volume')) columnMap['volume'] = index;
                if (lowerName.includes('delta')) columnMap['delta'] = index;
                if (lowerName.includes('theta')) columnMap['theta'] = index;
                if (lowerName.includes('days') && lowerName.includes('expir')) columnMap['days_to_expiry'] = index;
                if (lowerName.includes('net') && lowerName.includes('daily')) columnMap['net_daily_premium'] = index;
                if (lowerName.includes('net') && lowerName.includes('prem') && !lowerName.includes('daily')) columnMap['net_premium'] = index;
            });
        }
        
        // Get raw numeric value from a cell
        function getRawValue(cell) {
            const rawValue = cell.getAttribute('data-raw');
            if (rawValue !== null && rawValue !== '') {
                const num = parseFloat(rawValue);
                if (!isNaN(num)) return num;
            }
            // Fallback: try to parse from text
            const text = cell.textContent.trim();
            const num = parseFloat(text.replace(/[^0-9.-]/g, ''));
            return isNaN(num) ? null : num;
        }
        
        // Get raw text value from a cell
        function getRawText(cell) {
            const rawValue = cell.getAttribute('data-raw-text');
            if (rawValue !== null) return rawValue;
            return cell.textContent.trim();
        }
        
        // Evaluate a mathematical expression like "curr_price*1.05" or "strike_price+100"
        function evaluateMathExpression(expression, row, tableId) {
            // Extract field names from expression (alphanumeric + underscore)
            const fieldPattern = /[a-zA-Z_][a-zA-Z0-9_]*/g;
            const fields = expression.match(fieldPattern) || [];
            
            // Filter out JavaScript keywords and common math functions
            const jsKeywords = ['if', 'else', 'for', 'while', 'function', 'return', 'var', 'let', 'const', 'true', 'false', 'null', 'undefined'];
            const validFields = fields.filter(f => !jsKeywords.includes(f.toLowerCase()));
            
            // Get unique fields and their values
            const fieldValues = {};
            for (const field of validFields) {
                // Skip if we already processed this field
                if (fieldValues.hasOwnProperty(field)) continue;
                
                const colIndex = findColumnIndex(field, tableId);
                if (colIndex >= 0) {
                    const cell = row.cells[colIndex];
                    if (cell) {
                        const value = getRawValue(cell);
                        if (value !== null) {
                            fieldValues[field] = value;
                        } else {
                            console.warn('Field', field, 'found but has no numeric value');
                            return null; // Field not found or has no value
                        }
                    } else {
                        console.warn('Field', field, 'column index', colIndex, 'but cell not found');
                        return null; // Cell not found
                    }
                } else {
                    console.warn('Field', field, 'not found in table columns');
                    return null; // Field not found
                }
            }
            
            // Replace field names with their values (in reverse order of length to avoid partial replacements)
            let evalExpr = expression;
            const sortedFields = Object.keys(fieldValues).sort((a, b) => b.length - a.length);
            for (const field of sortedFields) {
                const value = fieldValues[field];
                // Replace field name, ensuring we match whole identifiers (not part of another identifier)
                // Escape special regex characters in the field name
                const escapedField = field.replace(/[.*+?^${}()|[\\]\\\\]/g, '\\\\$&');
                // Match field name as whole word (not part of another identifier)
                const regex = new RegExp('(^|[^a-zA-Z0-9_])' + escapedField + '([^a-zA-Z0-9_]|$)', 'g');
                evalExpr = evalExpr.replace(regex, function(match, before, after) {
                    // Preserve the before/after characters, replace only the field name
                    return before + value + after;
                });
            }
            
            // Safely evaluate the expression
            try {
                // Use Function constructor for safer evaluation
                // This evaluates basic math expressions with +, -, *, /
                const result = Function('"use strict"; return (' + evalExpr + ')')();
                return typeof result === 'number' && !isNaN(result) ? result : null;
            } catch (e) {
                console.error('Error evaluating math expression:', evalExpr, e);
                return null;
            }
        }
        
        // Get all available column names for error messages
        function getAllColumnNames(tableId) {
            const table = document.getElementById(tableId);
            // Return empty array if table doesn't exist (prevents null reference errors)
            if (!table) {
                return [];
            }
            // Only get headers from the column-header-row, not group headers
            const headerRow = table.querySelector('tr.column-header-row');
            const headers = headerRow ? Array.from(headerRow.querySelectorAll('th')) : [];
            const columnNames = [];
            headers.forEach(th => {
                const filterableName = th.getAttribute('data-filterable-name');
                if (filterableName) {
                    columnNames.push(filterableName);
                } else {
                    // Get text from the filterable name span if available, otherwise use display name
                    const filterableSpan = th.querySelector('.column-name-filterable');
                    const displaySpan = th.querySelector('.column-name-display');
                    if (filterableSpan && filterableSpan.textContent) {
                        columnNames.push(filterableSpan.textContent.trim());
                    } else if (displaySpan && displaySpan.textContent) {
                        columnNames.push(displaySpan.textContent.trim().replace(/\\s+/g, ' '));
                    } else {
                        const headerText = th.textContent.trim().replace(/\\s+/g, ' ').replace(/\\n/g, '');
                        if (headerText) {
                            columnNames.push(headerText);
                        }
                    }
                }
            });
            return columnNames;
        }
        
        // Find column index by field name (supports substring matching)
        function findColumnIndex(fieldName, tableId) {
            const table = document.getElementById(tableId);
            // Return -1 if table doesn't exist (prevents null reference errors)
            if (!table) {
                return -1;
            }
            // Only get headers from the column-header-row, not group headers
            const headerRow = table.querySelector('tr.column-header-row');
            const headers = headerRow ? Array.from(headerRow.querySelectorAll('th')) : [];
            const lowerField = fieldName.toLowerCase().replace(/\\s+/g, '_').trim();
            
            if (!lowerField) return -1;
            
            // First, try to match against data-filterable-name attribute (most reliable)
            for (let i = 0; i < headers.length; i++) {
                const filterableName = headers[i].getAttribute('data-filterable-name');
                if (filterableName) {
                    const normalizedFilterable = filterableName.toLowerCase().replace(/\\s+/g, '_').trim();
                    // Exact match (most reliable)
                    if (normalizedFilterable === lowerField) {
                        return i;
                    }
                }
            }
            
            // Try exact match against header text (normalized)
            for (let i = 0; i < headers.length; i++) {
                const headerText = headers[i].textContent.trim().toLowerCase().replace(/\\s+/g, '_').replace(/\\n/g, '').trim();
                if (headerText === lowerField) {
                    return i;
                }
            }
            
            // Try matching against data-filterable-name with normalization
            for (let i = 0; i < headers.length; i++) {
                const filterableName = headers[i].getAttribute('data-filterable-name');
                if (filterableName) {
                    const normalizedFilterable = filterableName.toLowerCase().replace(/\\s+/g, '_').trim();
                    // Check if field name is contained in filterable name or vice versa
                    // e.g., 'curr_price' should match 'current_price'
                    if (normalizedFilterable === lowerField || 
                        normalizedFilterable.replace('current', 'curr') === lowerField ||
                        lowerField.replace('curr', 'current') === normalizedFilterable) {
                        return i;
                    }
                }
            }
            
            // Try substring match on filterable name (but prefer exact matches)
            for (let i = 0; i < headers.length; i++) {
                const filterableName = headers[i].getAttribute('data-filterable-name');
                if (filterableName) {
                    const normalizedFilterable = filterableName.toLowerCase().replace(/\\s+/g, '_').trim();
                    // Only match if one contains the other (but not too loose)
                    if ((normalizedFilterable.length >= 3 && lowerField.length >= 3) &&
                        (normalizedFilterable.includes(lowerField) || lowerField.includes(normalizedFilterable))) {
                        return i;
                    }
                }
            }
            
            // Try column map
            if (columnMap[fieldName] !== undefined) {
                return columnMap[fieldName];
            }
            
            return -1;
        }
        
        // Parse market cap value with B/M/T suffixes
        function parseMarketCapValue(valueStr) {
            valueStr = valueStr.trim().toUpperCase();
            if (valueStr.endsWith('T')) {
                return parseFloat(valueStr.slice(0, -1)) * 1e12;
            } else if (valueStr.endsWith('B')) {
                return parseFloat(valueStr.slice(0, -1)) * 1e9;
            } else if (valueStr.endsWith('M')) {
                return parseFloat(valueStr.slice(0, -1)) * 1e6;
            }
            return parseFloat(valueStr);
        }
        
        // Parse filter expression
        function parseFilterExpression(expression, tableId) {
            expression = expression.trim();
            if (!expression) return null;
            
            // Handle exists/not_exists
            const existsMatch = expression.match(/^([a-zA-Z_][a-zA-Z0-9_]*)\\s+(exists|not_exists)$/i);
            if (existsMatch) {
                return {
                    field: existsMatch[1],
                    operator: existsMatch[2].toLowerCase(),
                    value: null,
                    isFieldComparison: false
                };
            }
            
            // Parse comparison operators
            const operators = ['>=', '<=', '==', '!=', '>', '<'];
            for (const op of operators) {
                if (expression.includes(op)) {
                    const parts = expression.split(op, 2);
                    if (parts.length === 2) {
                        const fieldExpr = parts[0].trim();
                        const valueStr = parts[1].trim();
                        
                        // Check for mathematical expressions in field FIRST
                        const hasMath = /[+\\-*/]/.test(fieldExpr);
                        if (hasMath) {
                            // Check if the value is also a field (field-to-field comparison with math)
                            // Only check if tableId is provided
                            if (tableId) {
                                const valueColIndex = findColumnIndex(valueStr, tableId);
                                if (valueColIndex >= 0) {
                                    // Math expression compared to another field: "curr_price*1.05 < strike_price"
                                    return {
                                        field: fieldExpr,
                                        operator: op,
                                        value: valueStr,
                                        isFieldComparison: true,
                                        hasMath: true
                                    };
                                }
                            }
                            // Math expression compared to a value: "curr_price*1.05 < 150"
                            let value = valueStr;
                            const numValue = parseFloat(valueStr);
                            if (!isNaN(numValue)) {
                                value = numValue;
                            }
                            return {
                                field: fieldExpr,
                                operator: op,
                                value: value,
                                isFieldComparison: false,
                                hasMath: true
                            };
                        }
                        
                        // Check if value is a field name (field-to-field comparison without math)
                        // Only check if tableId is provided
                        if (tableId) {
                            const valueColIndex = findColumnIndex(valueStr, tableId);
                            if (valueColIndex >= 0) {
                                return {
                                    field: fieldExpr,
                                    operator: op,
                                    value: valueStr,
                                    isFieldComparison: true
                                };
                            }
                        }
                        
                        // Regular value comparison
                        let value = valueStr;
                        // Try to parse as number
                        if (fieldExpr.toLowerCase().includes('market_cap') || fieldExpr.toLowerCase().includes('market cap')) {
                            value = parseMarketCapValue(valueStr);
                        } else {
                            const numValue = parseFloat(valueStr);
                            if (!isNaN(numValue)) {
                                value = numValue;
                            }
                        }
                        
                        return {
                            field: fieldExpr,
                            operator: op,
                            value: value,
                            isFieldComparison: false
                        };
                    }
                }
            }
            
            return null;
        }
        
        // Evaluate filter expression for a row
        function evaluateFilter(filter, row, tableId) {
            // Handle exists/not_exists (only works for single fields, not expressions)
            if (filter.operator === 'exists' || filter.operator === 'not_exists') {
                if (filter.hasMath) return false; // Can't use exists/not_exists with math expressions
                const colIndex = findColumnIndex(filter.field, tableId);
                if (colIndex < 0) {
                    console.warn('Column not found for filter:', filter.field);
                    return false;
                }
                const cell = row.cells[colIndex];
                if (!cell) return false;
                
                if (filter.operator === 'exists') {
                    const rawValue = cell.getAttribute('data-raw');
                    const text = cell.textContent.trim();
                    return (rawValue !== null && rawValue !== '') || (text !== '' && text !== 'N/A');
                }
                if (filter.operator === 'not_exists') {
                    const rawValue = cell.getAttribute('data-raw');
                    const text = cell.textContent.trim();
                    return (rawValue === null || rawValue === '') && (text === '' || text === 'N/A');
                }
            }
            
            // Handle mathematical expressions
            if (filter.hasMath) {
                // Evaluate the mathematical expression on the left side
                const leftValue = evaluateMathExpression(filter.field, row, tableId);
                if (leftValue === null) {
                    console.warn('Could not evaluate math expression:', filter.field);
                    return false; // Can't evaluate expression
                }
                
                // Get the right side value
                let rightValue = null;
                if (filter.isFieldComparison) {
                    // Right side is also a field: "curr_price*1.05 < strike_price"
                    const valueColIndex = findColumnIndex(filter.value, tableId);
                    if (valueColIndex < 0) {
                        console.warn('Column not found for math expression comparison value:', filter.value);
                        return false;
                    }
                    const valueCell = row.cells[valueColIndex];
                    if (!valueCell) return false;
                    rightValue = getRawValue(valueCell);
                } else {
                    // Right side is a numeric value: "curr_price*1.05 < 150"
                    rightValue = typeof filter.value === 'string' ? parseFloat(filter.value) : filter.value;
                }
                
                if (rightValue === null || isNaN(rightValue)) {
                    console.warn('Invalid right side value for math expression:', filter.value);
                    return false;
                }
                
                // Compare the evaluated expression with the right side
                switch (filter.operator) {
                    case '>': return leftValue > rightValue;
                    case '>=': return leftValue >= rightValue;
                    case '<': return leftValue < rightValue;
                    case '<=': return leftValue <= rightValue;
                    case '==': return Math.abs(leftValue - rightValue) < 0.0001;
                    case '!=': return Math.abs(leftValue - rightValue) >= 0.0001;
                    default: return false;
                }
            }
            
            // Handle field-to-field comparison (without math)
            if (filter.isFieldComparison) {
                const colIndex = findColumnIndex(filter.field, tableId);
                if (colIndex < 0) {
                    console.warn('Column not found for filter field:', filter.field);
                    return false;
                }
                const cell = row.cells[colIndex];
                if (!cell) return false;
                
                const valueColIndex = findColumnIndex(filter.value, tableId);
                if (valueColIndex < 0) {
                    console.warn('Column not found for filter comparison value:', filter.value);
                    return false;
                }
                const valueCell = row.cells[valueColIndex];
                if (!valueCell) return false;
                
                const cellValue = getRawValue(cell);
                const compareValue = getRawValue(valueCell);
                if (cellValue === null || compareValue === null) return false;
                
                switch (filter.operator) {
                    case '>': return cellValue > compareValue;
                    case '>=': return cellValue >= compareValue;
                    case '<': return cellValue < compareValue;
                    case '<=': return cellValue <= compareValue;
                    case '==': return Math.abs(cellValue - compareValue) < 0.0001;
                    case '!=': return Math.abs(cellValue - compareValue) >= 0.0001;
                    default: return false;
                }
            }
            
            // Handle value comparison (regular field to value)
            const colIndex = findColumnIndex(filter.field, tableId);
            if (colIndex < 0) {
                console.warn('Column not found for filter field:', filter.field);
                return false;
            }
            const cell = row.cells[colIndex];
            if (!cell) return false;
            
            const cellValue = getRawValue(cell);
            if (cellValue === null) return false;
            
            const filterValue = typeof filter.value === 'string' ? parseFloat(filter.value) : filter.value;
            if (isNaN(filterValue)) {
                // String comparison
                const cellText = getRawText(cell).toLowerCase();
                const filterText = String(filter.value).toLowerCase();
                switch (filter.operator) {
                    case '==': return cellText === filterText;
                    case '!=': return cellText !== filterText;
                    default: return false;
                }
            }
            
            // Numeric comparison
            switch (filter.operator) {
                case '>': return cellValue > filterValue;
                case '>=': return cellValue >= filterValue;
                case '<': return cellValue < filterValue;
                case '<=': return cellValue <= filterValue;
                case '==': return Math.abs(cellValue - filterValue) < 0.0001;
                case '!=': return Math.abs(cellValue - filterValue) >= 0.0001;
                default: return false;
            }
        }
        
        // Apply all filters to table
        function applyFilters(prefix) {
            const tableId = prefix + 'resultsTable';
            const table = document.getElementById(tableId);
            if (!table) return;
            
            const tbody = table.querySelector('tbody');
            const rows = Array.from(tbody.querySelectorAll('tr'));
            const errorDiv = document.getElementById(prefix + 'filterError');
            
            if (errorDiv) errorDiv.textContent = '';
            
            initStateForPrefix(prefix);
            
            if (activeFilters[prefix].length === 0) {
                rows.forEach(row => {
                    row.style.display = '';
                });
                updateVisibleCount(prefix);
                applyRowStriping(prefix);
                syncCardVisibility(prefix);
                return;
            }
            
            rows.forEach(row => {
                let matches = activeFilters[prefix].map(filter => evaluateFilter(filter, row, tableId));
                
                let shouldShow;
                if (filterLogic[prefix] === 'AND') {
                    shouldShow = matches.every(m => m);
                } else {
                    shouldShow = matches.some(m => m);
                }
                
                row.style.display = shouldShow ? '' : 'none';
            });
            
            updateVisibleCount(prefix);
            applyRowStriping(prefix);
            syncCardVisibility(prefix);
        }
        
        // Validate filter fields exist
        function validateFilter(filter, tableId) {
            const errorMessages = [];
            
            // Check main field
            if (filter.field && !filter.hasMath) {
                // For non-math expressions, check if field exists
                const colIndex = findColumnIndex(filter.field, tableId);
                if (colIndex < 0) {
                    const availableColumns = getAllColumnNames(tableId);
                    const suggestions = availableColumns.filter(col => 
                        col.toLowerCase().includes(filter.field.toLowerCase()) || 
                        filter.field.toLowerCase().includes(col.toLowerCase())
                    ).slice(0, 5);
                    
                    let errorMsg = `Column "${filter.field}" not found. `;
                    if (suggestions.length > 0) {
                        errorMsg += `Did you mean: ${suggestions.join(', ')}? `;
                    }
                    // Show all available columns (not just first 10) so user can see what's available
                    errorMsg += `Available columns: ${availableColumns.join(', ')}`;
                    errorMessages.push(errorMsg);
                }
            } else if (filter.hasMath) {
                // For math expressions, extract field names and validate them
                const fieldPattern = /[a-zA-Z_][a-zA-Z0-9_]*/g;
                const fields = filter.field.match(fieldPattern) || [];
                const jsKeywords = ['if', 'else', 'for', 'while', 'function', 'return', 'var', 'let', 'const', 'true', 'false', 'null', 'undefined'];
                const validFields = fields.filter(f => !jsKeywords.includes(f.toLowerCase()));
                
                for (const field of validFields) {
                    const colIndex = findColumnIndex(field, tableId);
                    if (colIndex < 0) {
                        errorMessages.push(`Field "${field}" in expression "${filter.field}" not found.`);
                    }
                }
            }
            
            // Check comparison field if it's a field-to-field comparison
            if (filter.isFieldComparison && filter.value) {
                const valueColIndex = findColumnIndex(filter.value, tableId);
                if (valueColIndex < 0) {
                    const availableColumns = getAllColumnNames(tableId);
                    const suggestions = availableColumns.filter(col => 
                        col.toLowerCase().includes(String(filter.value).toLowerCase()) || 
                        String(filter.value).toLowerCase().includes(col.toLowerCase())
                    ).slice(0, 5);
                    
                    let errorMsg = `Column "${filter.value}" not found for comparison. `;
                    if (suggestions.length > 0) {
                        errorMsg += `Did you mean: ${suggestions.join(', ')}? `;
                    }
                    // Show all available columns so user can see what's available
                    errorMsg += `Available columns: ${availableColumns.join(', ')}`;
                    errorMessages.push(errorMsg);
                }
            }
            
            return errorMessages;
        }
        
        // Add filter
        function addFilter(prefix) {
            const input = document.getElementById(prefix + 'filterInput');
            const expression = input.value.trim();
            const errorDiv = document.getElementById(prefix + 'filterError');
            const tableId = prefix + 'resultsTable';
            
            if (errorDiv) errorDiv.textContent = '';
            
            if (!expression) return;
            
            initStateForPrefix(prefix);
            
            const filter = parseFilterExpression(expression, tableId);
            if (!filter) {
                if (errorDiv) {
                    errorDiv.textContent = 'Invalid filter expression. Format: field operator value (e.g., "pe_ratio > 20", "volume exists")';
                }
                return;
            }
            
            // Validate filter fields exist
            const validationErrors = validateFilter(filter, tableId);
            if (validationErrors.length > 0) {
                if (errorDiv) {
                    errorDiv.textContent = validationErrors.join(' ');
                    errorDiv.style.display = 'block';
                    errorDiv.style.color = '#dc3545';
                    errorDiv.style.backgroundColor = '#f8d7da';
                }
                console.error('Filter validation errors:', validationErrors);
                return;
            }
            
            // Clear any previous errors if validation passes
            if (errorDiv) {
                errorDiv.textContent = '';
                errorDiv.style.display = 'none';
            }
            
            activeFilters[prefix].push(filter);
            input.value = '';
            
            // Update filter display
            updateFilterDisplay(prefix);
            applyFilters(prefix);
            updateURL();
        }
        
        // Remove filter
        function removeFilter(prefix, index) {
            initStateForPrefix(prefix);
            activeFilters[prefix].splice(index, 1);
            updateFilterDisplay(prefix);
            applyFilters(prefix);
            updateURL();
        }
        
        // Clear all filters
        function clearFilters(prefix) {
            initStateForPrefix(prefix);
            activeFilters[prefix] = [];
            updateFilterDisplay(prefix);
            applyFilters(prefix);
            updateURL();
        }
        
        // Update filter display
        function updateFilterDisplay(prefix) {
            const container = document.getElementById(prefix + 'activeFilters');
            if (!container) return;
            
            initStateForPrefix(prefix);
            container.innerHTML = '';
            
            if (activeFilters[prefix].length === 0) {
                container.innerHTML = '<p style="color: #6c757d; font-style: italic;">No active filters</p>';
                return;
            }
            
            activeFilters[prefix].forEach((filter, index) => {
                const filterDiv = document.createElement('div');
                filterDiv.style.cssText = 'display: inline-block; margin: 5px; padding: 5px 10px; background: #667eea; color: white; border-radius: 5px; font-size: 0.9em;';
                
                const filterText = document.createElement('span');
                filterText.textContent = `${filter.field} ${filter.operator} ${filter.value !== null ? filter.value : ''}`;
                filterDiv.appendChild(filterText);
                
                const removeBtn = document.createElement('button');
                removeBtn.textContent = '×';
                removeBtn.style.cssText = 'margin-left: 8px; background: rgba(255,255,255,0.3); border: none; color: white; cursor: pointer; border-radius: 3px; padding: 2px 6px;';
                removeBtn.onclick = () => removeFilter(prefix, index);
                filterDiv.appendChild(removeBtn);
                
                container.appendChild(filterDiv);
            });
        }
        
        // Update filter logic
        function updateFilterLogic(prefix, logic) {
            initStateForPrefix(prefix);
            filterLogic[prefix] = logic;
            applyFilters(prefix);
            updateURL();
        }
        
        // Update visible count
        function updateVisibleCount(prefix) {
            const tableId = prefix + 'resultsTable';
            const table = document.getElementById(tableId);
            if (!table) return;
            const tbody = table.querySelector('tbody');
            const visibleRows = Array.from(tbody.querySelectorAll('tr')).filter(row => row.style.display !== 'none');
            const visibleCountEl = document.getElementById(prefix + 'visibleCount');
            if (visibleCountEl) {
                visibleCountEl.textContent = visibleRows.length;
            }
        }
        
        // Apply row striping based on visible rows
        function applyRowStriping(prefix) {
            const tableId = prefix + 'resultsTable';
            const table = document.getElementById(tableId);
            if (!table) return;
            const tbody = table.querySelector('tbody');
            const rows = Array.from(tbody.querySelectorAll('tr'));
            
            // Remove all even-row classes first
            rows.forEach(row => row.classList.remove('even-row'));
            
            // Apply even-row class to every other visible row
            let visibleIndex = 0;
            rows.forEach(row => {
                if (row.style.display !== 'none') {
                    if (visibleIndex % 2 === 1) {
                        row.classList.add('even-row');
                    }
                    visibleIndex++;
                }
            });
        }
        
        // Apply column striping based on visible columns
        function applyColumnStriping(prefix) {
            const tableId = prefix + 'resultsTable';
            const table = document.getElementById(tableId);
            if (!table) return;
            const thead = table.querySelector('thead');
            const tbody = table.querySelector('tbody');
            const wrapperId = prefix + 'tableWrapper';
            const wrapper = document.getElementById(wrapperId);
            
            if (!thead || !tbody) return;
            
            // Only use column headers from the column-header-row, not group headers
            const headerRow = thead.querySelector('tr.column-header-row');
            const headers = headerRow ? Array.from(headerRow.querySelectorAll('th')) : Array.from(thead.querySelectorAll('th'));
            const rows = Array.from(tbody.querySelectorAll('tr'));
            
            // Check if hidden columns are currently hidden
            const hideHidden = wrapper && wrapper.classList.contains('hide-hidden');
            
            // Remove all even-col classes first
            headers.forEach(th => th.classList.remove('even-col'));
            rows.forEach(row => {
                Array.from(row.querySelectorAll('td')).forEach(td => td.classList.remove('even-col'));
            });
            
            // Find visible column indices (columns that are actually displayed)
            const visibleColumnIndices = [];
            headers.forEach((th, index) => {
                // Column is visible if:
                // 1. It doesn't have always-hidden class (always-hidden columns are never visible)
                // 2. It doesn't have is-hidden-col class, OR
                // 3. It has is-hidden-col but hide-hidden is false (showing all columns)
                const isAlwaysHidden = th.classList.contains('always-hidden');
                const isHiddenCol = th.classList.contains('is-hidden-col');
                if (!isAlwaysHidden && (!isHiddenCol || !hideHidden)) {
                    visibleColumnIndices.push(index);
                }
            });
            
            // Apply even-col class to every other visible column
            visibleColumnIndices.forEach((colIndex, visibleIndex) => {
                if (visibleIndex % 2 === 1) {
                    // Add to header
                    if (headers[colIndex]) {
                        headers[colIndex].classList.add('even-col');
                    }
                    // Add to all cells in this column
                    rows.forEach(row => {
                        const cell = row.cells[colIndex];
                        if (cell) {
                            cell.classList.add('even-col');
                        }
                    });
                }
            });
        }
        
        function sortTable(prefix, columnIndex) {
            const tableId = prefix + 'resultsTable';
            const table = document.getElementById(tableId);
            if (!table) return;
            const tbody = table.querySelector('tbody');
            const rows = Array.from(tbody.querySelectorAll('tr:not([style*="display: none"])'));
            // Only select headers from the column-header-row (second row), not group headers
            const headerRow = table.querySelector('tr.column-header-row');
            const headers = headerRow ? headerRow.querySelectorAll('th') : table.querySelectorAll('th');
            
            initStateForPrefix(prefix);
            
            // Remove sort classes from all column headers
            headers.forEach(h => {
                h.classList.remove('sort-asc', 'sort-desc');
            });
            
            // Determine sort direction
            if (currentSortColumn[prefix] === columnIndex) {
                sortDirection[prefix][columnIndex] = sortDirection[prefix][columnIndex] === 'asc' ? 'desc' : 'asc';
            } else {
                if (!sortDirection[prefix]) sortDirection[prefix] = {};
                sortDirection[prefix][columnIndex] = 'asc';
                currentSortColumn[prefix] = columnIndex;
            }
            
            // Add sort class to current header
            if (headers[columnIndex]) {
                headers[columnIndex].classList.add(sortDirection[prefix][columnIndex] === 'asc' ? 'sort-asc' : 'sort-desc');
            }
            
            // Check if sorting by change_pct (renamed from price_with_change)
            // We can sort directly using the data-raw attribute which contains the percentage value
            // No need to find a separate column - the change_pct column itself has the percentage in data-raw
            const currentHeader = headers[columnIndex];
            const filterableName = currentHeader.getAttribute('data-filterable-name');
            let sortColumnIndex = columnIndex;
            // For change_pct column, we'll use its own data-raw attribute which contains the percentage
            // The JavaScript sorting code below will automatically use data-raw if available
            
            // Check if this is a date column
            // Note: latest_option_writets is NOT a date column - it's age in seconds (numeric)
            const filterableLower = filterableName ? filterableName.toLowerCase() : '';
            const isDateColumn = filterableLower && (
                filterableLower.includes('expiration_date') ||
                filterableLower.includes('exp_date') ||
                (filterableLower.endsWith('_expiration_date') && !filterableLower.includes('latest_option_writets') && !filterableLower.includes('latest_opt_ts')) ||
                (filterableLower.endsWith('_exp_date') && !filterableLower.includes('latest_option_writets') && !filterableLower.includes('latest_opt_ts')) ||
                filterableLower === 'l_expiration_date' ||
                filterableLower === 'long_expiration_date' ||
                filterableLower === 'expiration_date'
            );
            
            // Check if this is an age column (latest_option_writets - age in seconds)
            const isAgeColumn = filterableLower && (
                filterableLower.includes('latest_option_writets') ||
                filterableLower.includes('latest_opt_ts') ||
                filterableLower.endsWith('_writets')
            );
            
            // Sort rows
            rows.sort((a, b) => {
                const aCell = a.cells[sortColumnIndex];
                const bCell = b.cells[sortColumnIndex];
                if (!aCell || !bCell) return 0;
                
                const aRaw = aCell.getAttribute('data-raw');
                const bRaw = bCell.getAttribute('data-raw');
                
                let comparison = 0;
                
                // Handle age columns (latest_option_writets) - sort as numeric (smaller = more recent)
                if (isAgeColumn) {
                    // Age in seconds - smaller value = more recent
                    let aNum = aRaw !== null ? parseFloat(aRaw) : NaN;
                    let bNum = bRaw !== null ? parseFloat(bRaw) : NaN;
                    
                    if (isNaN(aNum)) {
                        aNum = parseFloat(aCell.textContent.trim().replace(/[^0-9.-]/g, ''));
                    }
                    if (isNaN(bNum)) {
                        bNum = parseFloat(bCell.textContent.trim().replace(/[^0-9.-]/g, ''));
                    }
                    
                    if (!isNaN(aNum) && !isNaN(bNum)) {
                        comparison = aNum - bNum;  // Smaller age = more recent
                    } else if (isNaN(aNum) && !isNaN(bNum)) {
                        comparison = 1; // a comes after b
                    } else if (!isNaN(aNum) && isNaN(bNum)) {
                        comparison = -1; // a comes before b
                    } else {
                        // Both are NaN, use text comparison
                        const aText = aCell.textContent.trim();
                        const bText = bCell.textContent.trim();
                        comparison = aText.localeCompare(bText);
                    }
                }
                // Handle date columns specially
                else if (isDateColumn) {
                    // For date columns, data-raw contains timestamp in milliseconds
                    let aTimestamp = aRaw !== null ? parseFloat(aRaw) : NaN;
                    let bTimestamp = bRaw !== null ? parseFloat(bRaw) : NaN;
                    
                    // If timestamp parsing fails, try to parse from text (YYYY-MM-DD format)
                    if (isNaN(aTimestamp)) {
                        const aText = aCell.textContent.trim();
                        const aDate = new Date(aText);
                        aTimestamp = isNaN(aDate.getTime()) ? NaN : aDate.getTime();
                    }
                    if (isNaN(bTimestamp)) {
                        const bText = bCell.textContent.trim();
                        const bDate = new Date(bText);
                        bTimestamp = isNaN(bDate.getTime()) ? NaN : bDate.getTime();
                    }
                    
                    if (!isNaN(aTimestamp) && !isNaN(bTimestamp)) {
                        comparison = aTimestamp - bTimestamp;
                    } else if (isNaN(aTimestamp) && !isNaN(bTimestamp)) {
                        comparison = 1; // a comes after b
                    } else if (!isNaN(aTimestamp) && isNaN(bTimestamp)) {
                        comparison = -1; // a comes before b
                    } else {
                        // Both are NaN, use text comparison
                        const aText = aCell.textContent.trim();
                        const bText = bCell.textContent.trim();
                        comparison = aText.localeCompare(bText);
                    }
                } else {
                    // Regular numeric/text sorting
                    let aNum = aRaw !== null ? parseFloat(aRaw) : NaN;
                    let bNum = bRaw !== null ? parseFloat(bRaw) : NaN;
                    
                    if (isNaN(aNum)) {
                        aNum = parseFloat(aCell.textContent.trim().replace(/[^0-9.-]/g, ''));
                    }
                    if (isNaN(bNum)) {
                        bNum = parseFloat(bCell.textContent.trim().replace(/[^0-9.-]/g, ''));
                    }
                    
                    if (!isNaN(aNum) && !isNaN(bNum)) {
                        comparison = aNum - bNum;
                    } else {
                        const aText = aCell.textContent.trim();
                        const bText = bCell.textContent.trim();
                        comparison = aText.localeCompare(bText);
                    }
                }
                
                return sortDirection[prefix][columnIndex] === 'asc' ? comparison : -comparison;
            });
            
            // Re-append sorted rows (only visible ones)
            const hiddenRows = Array.from(tbody.querySelectorAll('tr[style*="display: none"]'));
            rows.forEach(row => tbody.appendChild(row));
            hiddenRows.forEach(row => tbody.appendChild(row));
            
            // Sync card order with table order
            syncCardOrder(prefix);
            
            updateVisibleCount(prefix);
            applyRowStriping(prefix);
        }
        
        // Sync card order with table row order
        function syncCardOrder(prefix) {
            const tableId = prefix + 'resultsTable';
            const table = document.getElementById(tableId);
            const tbody = table ? table.querySelector('tbody') : null;
            const cardWrapperId = prefix + 'cardWrapper';
            const cardWrapper = document.getElementById(cardWrapperId);
            if (!tbody || !cardWrapper) return;
            
            const rows = Array.from(tbody.querySelectorAll('tr'));
            const cards = cardWrapper ? Array.from(cardWrapper.querySelectorAll('.data-card')) : [];
            
            // Create a map of row index to card
            const cardMap = new Map();
            cards.forEach(card => {
                const rowIndex = parseInt(card.getAttribute('data-row-index'));
                if (!isNaN(rowIndex)) {
                    cardMap.set(rowIndex, card);
                }
            });
            
            // Reorder cards to match table row order
            rows.forEach((row, index) => {
                const rowIndex = parseInt(row.getAttribute('data-row-index') || index);
                const card = cardMap.get(rowIndex);
                if (card) {
                    cardWrapper.appendChild(card);
                }
            });
        }
        
        // Tab switching functionality
        function switchTab(tabIndex) {
            const tabContents = document.querySelectorAll('.tab-content');
            tabContents.forEach(content => {
                content.classList.remove('active');
                content.style.display = 'none';
            });
            
            const tabButtons = document.querySelectorAll('.tab-button');
            tabButtons.forEach(button => button.classList.remove('active'));
            
            if (tabContents[tabIndex]) {
                tabContents[tabIndex].classList.add('active');
                tabContents[tabIndex].style.display = 'block';
            }
            if (tabButtons[tabIndex]) {
                tabButtons[tabIndex].classList.add('active');
            }
        }
        
        // Handle Enter key in filter input
        function handleFilterKeyPress(event, prefix) {
            if (event.key === 'Enter') {
                addFilter(prefix);
            }
        }
        
        // Toggle filter section and column names
        function toggleFilterSection(prefix) {
            const filterSection = document.getElementById(prefix + 'filterSection');
            const button = document.getElementById(prefix + 'toggleFilterBtn');
            const tableId = prefix + 'resultsTable';
            // Only toggle filterable display on column headers, not group headers
            const headerRow = document.querySelector('#' + tableId + ' tr.column-header-row');
            const headers = headerRow ? headerRow.querySelectorAll('th') : document.querySelectorAll('#' + tableId + ' th');
            
            if (!filterSection || !button) return;
            
            // Toggle filter section
            const isExpanded = filterSection.classList.contains('expanded');
            
            if (isExpanded) {
                // Collapse: hide filter section and show display names (or short/long for grouped columns)
                filterSection.classList.remove('expanded');
                headers.forEach(th => {
                    th.classList.remove('showing-filterable');
                });
                button.textContent = '🔍 Filter';
                button.title = 'Show filter options and filterable column names';
            } else {
                // Expand: show filter section and show filterable names
                filterSection.classList.add('expanded');
                headers.forEach(th => {
                    th.classList.add('showing-filterable');
                });
                button.textContent = '✖️ Hide Filter';
                button.title = 'Hide filter options and show display column names';
            }
        }

        // Toggle hidden columns visibility
        function toggleHiddenColumns(prefix) {
            const wrapperId = prefix + 'tableWrapper';
            const wrapper = document.getElementById(wrapperId);
            const btn = document.getElementById(prefix + 'toggleHiddenBtn');
            if (!wrapper || !btn) return;
            const willShow = wrapper.classList.contains('hide-hidden');
            if (willShow) {
                wrapper.classList.remove('hide-hidden');
                btn.textContent = '🙈 Hide hidden columns';
                btn.title = 'Hide the default-hidden columns';
            } else {
                wrapper.classList.add('hide-hidden');
                btn.textContent = '👁️ Show hidden columns';
                btn.title = 'Show the default-hidden columns';
            }
            // Reapply striping after column visibility changes
            applyRowStriping(prefix);
            applyColumnStriping(prefix);
        }
        
        // Update URL with current filters
        function updateURL() {
            const params = new URLSearchParams();
            
            // Add filter logic for calls (only if not default AND)
            if (filterLogic['calls'] && filterLogic['calls'] !== 'AND') {
                params.set('calls_filterLogic', filterLogic['calls']);
            }
            
            // Add filters for calls
            if (activeFilters['calls'] && activeFilters['calls'].length > 0) {
                const filterStrings = activeFilters['calls'].map(f => {
                    let filterStr = f.field;
                    if (f.operator) {
                        filterStr += ' ' + f.operator;
                        if (f.value !== null) {
                            filterStr += ' ' + f.value;
                        }
                    }
                    return filterStr;
                });
                params.set('calls_filters', filterStrings.join('|'));
            }
            
            // Add filter logic for puts (only if not default AND)
            if (filterLogic['puts'] && filterLogic['puts'] !== 'AND') {
                params.set('puts_filterLogic', filterLogic['puts']);
            }
            
            // Add filters for puts
            if (activeFilters['puts'] && activeFilters['puts'].length > 0) {
                const filterStrings = activeFilters['puts'].map(f => {
                    let filterStr = f.field;
                    if (f.operator) {
                        filterStr += ' ' + f.operator;
                        if (f.value !== null) {
                            filterStr += ' ' + f.value;
                        }
                    }
                    return filterStr;
                });
                params.set('puts_filters', filterStrings.join('|'));
            }
            
            // Update URL without reloading page
            const newURL = window.location.pathname + (params.toString() ? '?' + params.toString() : '');
            window.history.replaceState({}, '', newURL);
        }
        
        // Load filters from URL
        function loadFiltersFromURL() {
            const params = new URLSearchParams(window.location.search);
            
            // Load filters for calls
            loadFiltersForPrefix('calls', params);
            
            // Load filters for puts
            loadFiltersForPrefix('puts', params);
        }
        
        // Helper function to load filters for a specific prefix
        function loadFiltersForPrefix(prefix, params) {
            const errorDiv = document.getElementById(prefix + 'filterError');
            if (errorDiv) {
                errorDiv.textContent = '';
                errorDiv.style.display = 'none';
            }
            
            initStateForPrefix(prefix);
            const tableId = prefix + 'resultsTable';
            
            // Load filter logic
            const urlFilterLogic = params.get(prefix + '_filterLogic');
            if (urlFilterLogic && (urlFilterLogic === 'AND' || urlFilterLogic === 'OR')) {
                filterLogic[prefix] = urlFilterLogic;
                // Update radio button
                const radio = document.querySelector(`input[name="${prefix}filterLogic"][value="${filterLogic[prefix]}"]`);
                if (radio) {
                    radio.checked = true;
                }
            }
            
            // Load filters
            const filtersParam = params.get(prefix + '_filters');
            if (filtersParam) {
                const filterStrings = filtersParam.split('|');
                activeFilters[prefix] = [];
                const validationErrors = [];
                
                for (const filterStr of filterStrings) {
                    if (filterStr.trim()) {
                        const filter = parseFilterExpression(filterStr.trim(), tableId);
                        if (filter) {
                            // Validate filter before adding
                            const errors = validateFilter(filter, tableId);
                            if (errors.length > 0) {
                                validationErrors.push(...errors);
                            } else {
                                activeFilters[prefix].push(filter);
                            }
                        }
                    }
                }
                
                // Show validation errors if any
                if (validationErrors.length > 0 && errorDiv) {
                    errorDiv.textContent = 'Warning: Some filters from URL could not be loaded: ' + validationErrors.join(' ');
                    errorDiv.style.display = 'block';
                }
                
                // If filters were loaded from URL, expand the filter section
                if (activeFilters[prefix].length > 0) {
                    const filterSection = document.getElementById(prefix + 'filterSection');
                    const button = document.getElementById(prefix + 'toggleFilterBtn');
                    // Only show filterable names on column headers, not group headers
                    const headerRow = document.querySelector('#' + tableId + ' tr.column-header-row');
                    const headers = headerRow ? headerRow.querySelectorAll('th') : document.querySelectorAll('#' + tableId + ' th');
                    
                    if (filterSection && button) {
                        filterSection.classList.add('expanded');
                        headers.forEach(th => {
                            th.classList.add('showing-filterable');
                        });
                        button.textContent = '✖️ Hide Filter';
                        button.title = 'Hide filter options and show display column names';
                    }
                }
                
                updateFilterDisplay(prefix);
                applyFilters(prefix);
            }
        }
        
        // Calculate and display time ago
        function updateTimeAgo() {
            const timeAgoEl = document.getElementById('timeAgo');
            const generatedTimeEl = document.getElementById('generatedTime');
            if (!timeAgoEl || !generatedTimeEl) return;
            
            // Get the ISO timestamp from the data-generated attribute on the paragraph
            const generatedTimeStr = generatedTimeEl.getAttribute('data-generated');
            if (!generatedTimeStr || generatedTimeStr.trim() === '' || generatedTimeStr === '+') {
                // Silently return if no valid timestamp
                return;
            }
            
            try {
                // Clean up the timestamp string - handle timezone offsets properly
                let cleanTimestamp = generatedTimeStr.trim();
                // If it's just a "+" or malformed, skip
                if (cleanTimestamp === '+' || cleanTimestamp.length < 10) {
                    return;
                }
                
                const generatedTime = new Date(cleanTimestamp);
                // Check if date is valid
                if (isNaN(generatedTime.getTime())) {
                    // Silently fail instead of logging error for malformed dates
                    timeAgoEl.textContent = '';
                    return;
                }
                
                const now = new Date();
                const diffMs = now - generatedTime;
                
                // Handle negative differences (future dates) or invalid calculations
                if (isNaN(diffMs) || diffMs < 0) {
                    timeAgoEl.textContent = '';
                    return;
                }
                
                const diffSeconds = Math.floor(diffMs / 1000);
                const diffMinutes = Math.floor(diffSeconds / 60);
                const diffHours = Math.floor(diffMinutes / 60);
                const diffDays = Math.floor(diffHours / 24);
                
                let timeAgoText = '';
                if (diffSeconds < 60) {
                    timeAgoText = `(${diffSeconds} second${diffSeconds !== 1 ? 's' : ''} ago)`;
                } else if (diffMinutes < 60) {
                    timeAgoText = `(${diffMinutes} minute${diffMinutes !== 1 ? 's' : ''} ago)`;
                } else if (diffHours < 24) {
                    timeAgoText = `(${diffHours} hour${diffHours !== 1 ? 's' : ''} ago)`;
                } else if (diffDays < 7) {
                    timeAgoText = `(${diffDays} day${diffDays !== 1 ? 's' : ''} ago)`;
                } else {
                    const diffWeeks = Math.floor(diffDays / 7);
                    // Validate diffWeeks is a valid number
                    if (isNaN(diffWeeks) || diffWeeks < 0) {
                        timeAgoEl.textContent = '';
                        return;
                    }
                    timeAgoText = `(${diffWeeks} week${diffWeeks !== 1 ? 's' : ''} ago)`;
                }
                
                timeAgoEl.textContent = timeAgoText;
            } catch (e) {
                console.error('Error calculating time ago:', e);
                timeAgoEl.textContent = '';
            }
        }
        
        // Toggle card details (expand/collapse)
        function toggleCardDetails(prefix, rowIndex) {
            const detailsEl = document.getElementById(prefix + 'cardDetails_' + rowIndex);
            const toggleBtn = document.getElementById(prefix + 'cardToggle_' + rowIndex);
            if (!detailsEl || !toggleBtn) return;
            
            const isExpanded = detailsEl.classList.contains('expanded');
            if (isExpanded) {
                detailsEl.classList.remove('expanded');
                toggleBtn.classList.remove('expanded');
                toggleBtn.querySelector('span:first-child').textContent = 'Show More Details';
            } else {
                detailsEl.classList.add('expanded');
                toggleBtn.classList.add('expanded');
                toggleBtn.querySelector('span:first-child').textContent = 'Hide Details';
            }
        }
        
        // Sync card visibility with table row visibility (for filtering)
        function syncCardVisibility(prefix) {
            const tableId = prefix + 'resultsTable';
            const table = document.getElementById(tableId);
            const tbody = table ? table.querySelector('tbody') : null;
            if (!tbody) return;
            
            const rows = Array.from(tbody.querySelectorAll('tr'));
            const cardWrapperId = prefix + 'cardWrapper';
            const cardWrapper = document.getElementById(cardWrapperId);
            const cards = cardWrapper ? Array.from(cardWrapper.querySelectorAll('.data-card')) : [];
            
            // Create a map of row index to card
            const cardMap = new Map();
            cards.forEach(card => {
                const rowIndex = card.getAttribute('data-row-index');
                if (rowIndex !== null) {
                    cardMap.set(rowIndex, card);
                }
            });
            
            // Sync visibility based on row index
            rows.forEach((row) => {
                const rowIndex = row.getAttribute('data-row-index');
                if (rowIndex !== null) {
                    const card = cardMap.get(rowIndex);
                    if (card) {
                        const isVisible = row.style.display !== 'none';
                        if (isVisible) {
                            card.classList.remove('hidden');
                        } else {
                            card.classList.add('hidden');
                        }
                    }
                }
            });
            
            // Update visible count for cards
            const visibleCards = cards.filter(card => !card.classList.contains('hidden'));
            const visibleCountEl = document.getElementById(prefix + 'visibleCount');
            if (visibleCountEl) {
                visibleCountEl.textContent = visibleCards.length;
            }
        }
        
        // Initialize
        document.addEventListener('DOMContentLoaded', function() {
            // Initialize state for both prefixes
            initStateForPrefix('calls');
            initStateForPrefix('puts');
            
            switchTab(0);
            
            // Initialize column maps for both tables
            initColumnMap('calls');
            initColumnMap('puts');
            
            // Update filter displays
            updateFilterDisplay('calls');
            updateFilterDisplay('puts');
            
            // Load filters from URL
            loadFiltersFromURL();
            
            updateTimeAgo(); // Update time ago on page load
            // Update time ago every minute
            setInterval(updateTimeAgo, 60000);
            
            // Ensure hidden columns are hidden on load for both tables
            ['calls', 'puts'].forEach(prefix => {
                const wrapper = document.getElementById(prefix + 'tableWrapper');
                const toggleBtn = document.getElementById(prefix + 'toggleHiddenBtn');
                const ensureHidden = wrapper && !wrapper.classList.contains('hide-hidden');
                if (ensureHidden) {
                    wrapper.classList.add('hide-hidden');
                }
                if (toggleBtn) {
                    if (wrapper && wrapper.classList.contains('hide-hidden')) {
                        toggleBtn.textContent = '👁️ Show hidden columns';
                        toggleBtn.title = 'Show the default-hidden columns';
                    } else {
                        toggleBtn.textContent = '🙈 Hide hidden columns';
                        toggleBtn.title = 'Hide the default-hidden columns';
                    }
                }
                
                // Apply initial row and column striping
                applyRowStriping(prefix);
                applyColumnStriping(prefix);
                
                // Sync card visibility on load
                syncCardVisibility(prefix);
                
                // Default sort by net_premium descending
                const netPremiumColIndex = findColumnIndex('net_premium', prefix + 'resultsTable');
                if (netPremiumColIndex >= 0) {
                    // Call sortTable twice: first sets to asc, second toggles to desc
                    sortTable(prefix, netPremiumColIndex);
                    sortTable(prefix, netPremiumColIndex);
                }
            });
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
    
    # Safely convert spread_impact_pct to numeric, handling malformed strings
    def safe_extract_number(val):
        if pd.isna(val):
            return np.nan
        try:
            val_str = str(val)
            try:
                return float(val_str)
            except (ValueError, TypeError):
                match = re.search(r'-?\d+\.?\d*', val_str)
                if match:
                    return float(match.group())
                return np.nan
        except (ValueError, TypeError, AttributeError):
            return np.nan
    
    # Ensure spread_impact_pct is numeric before calculations
    spread_impact_numeric = df['spread_impact_pct'].apply(safe_extract_number)
    spread_impact_numeric = pd.to_numeric(spread_impact_numeric, errors='coerce')
    
    # Spread Impact Statistics
    spread_impact_mean = spread_impact_numeric.mean()
    spread_impact_median = spread_impact_numeric.median()
    low_impact_count = (spread_impact_numeric < 5).sum()
    high_impact_count = (spread_impact_numeric > 10).sum()
    
    html_parts.append('                <div class="analysis-section">\n')
    html_parts.append('                    <h4>Spread Impact Statistics</h4>\n')
    html_parts.append(f'                    <p><span class="label">Average spread impact:</span> {spread_impact_mean:.2f}%</p>\n')
    html_parts.append(f'                    <p><span class="label">Median spread impact:</span> {spread_impact_median:.2f}%</p>\n')
    html_parts.append(f'                    <p><span class="label">Trades with &lt;5% impact:</span> {low_impact_count}/{len(df)}</p>\n')
    html_parts.append(f'                    <p><span class="label">Trades with &gt;10% impact:</span> <span class="risk-badge risk-high">{high_impact_count}/{len(df)} ⚠️</span></p>\n')
    html_parts.append('                </div>\n')
    
    # Liquidity Distribution
    if 'liquidity_score' in df.columns:
        # Safely convert liquidity_score to numeric
        liquidity_score_numeric = df['liquidity_score'].apply(safe_extract_number)
        liquidity_score_numeric = pd.to_numeric(liquidity_score_numeric, errors='coerce')
        
        high_liquidity = (liquidity_score_numeric >= 7).sum()
        medium_liquidity = ((liquidity_score_numeric >= 4) & (liquidity_score_numeric < 7)).sum()
        low_liquidity = (liquidity_score_numeric < 4).sum()
        
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
    # Convert net_daily_premi to numeric if it exists, handling object dtype
    if 'net_daily_premi' in df.columns:
        # Create a temporary numeric column for sorting
        df = df.copy()
        # Convert to string first, then clean and convert to numeric
        # This handles cases where values might be concatenated or malformed
        def safe_convert_to_numeric(val):
            if pd.isna(val):
                return np.nan
            try:
                # Convert to string and try to extract first valid number
                val_str = str(val)
                # Try direct conversion first
                try:
                    return float(val_str)
                except (ValueError, TypeError):
                    # If that fails, try to extract first number from string
                    # Find first number (including decimals)
                    match = re.search(r'-?\d+\.?\d*', val_str)
                    if match:
                        return float(match.group())
                    return np.nan
            except (ValueError, TypeError, AttributeError):
                return np.nan
        
        df['_net_daily_premi_numeric'] = df['net_daily_premi'].apply(safe_convert_to_numeric)
        # Use the numeric column for nlargest, then drop it
        # Filter out rows where the numeric value is NaN before sorting
        df_valid = df[df['_net_daily_premi_numeric'].notna()]
        if len(df_valid) > 0:
            top_10 = df_valid.nlargest(10, '_net_daily_premi_numeric').drop(columns=['_net_daily_premi_numeric'])
        else:
            # If no valid values, just take first 10 rows
            top_10 = df.head(10).drop(columns=['_net_daily_premi_numeric'], errors='ignore')
    elif 'net_daily_premium' in df.columns:
        # Try alternative column name
        df = df.copy()
        def safe_convert_to_numeric(val):
            if pd.isna(val):
                return np.nan
            try:
                val_str = str(val)
                try:
                    return float(val_str)
                except (ValueError, TypeError):
                    match = re.search(r'-?\d+\.?\d*', val_str)
                    if match:
                        return float(match.group())
                    return np.nan
            except (ValueError, TypeError, AttributeError):
                return np.nan
        
        df['_net_daily_premi_numeric'] = df['net_daily_premium'].apply(safe_convert_to_numeric)
        df_valid = df[df['_net_daily_premi_numeric'].notna()]
        if len(df_valid) > 0:
            top_10 = df_valid.nlargest(10, '_net_daily_premi_numeric').drop(columns=['_net_daily_premi_numeric'])
        else:
            top_10 = df.head(10).drop(columns=['_net_daily_premi_numeric'], errors='ignore')
    else:
        # Fallback: use first 10 rows if column doesn't exist
        top_10 = df.head(10)
    
    html_parts = []
    html_parts.append('        <div class="detailed-analysis">\n')
    html_parts.append('            <h2>📊 COMPREHENSIVE ANALYSIS: TOP 10 PICKS WITH OPTION TICKERS</h2>\n')
    
    # Add summary statistics at the top
    html_parts.append(generate_summary_statistics_html(df))
    
    # Helper function to safely get numeric value from row
    def safe_get_numeric(row, key, default=0):
        """Safely get and convert a numeric value from a row."""
        val = row.get(key, default)
        if pd.isna(val):
            return default
        try:
            # Handle malformed strings
            if isinstance(val, str):
                match = re.search(r'-?\d+\.?\d*', val)
                if match:
                    return float(match.group())
                return default
            return float(val)
        except (ValueError, TypeError):
            return default
    
    for idx, (row_idx, row) in enumerate(top_10.iterrows(), 1):
        ticker = row['ticker']
        
        # Get option type for this row
        option_type = str(row.get('option_type', 'call')).lower() if pd.notna(row.get('option_type')) else 'call'
        is_put = (option_type == 'put')
        
        # Calculate values - use helper function for safe conversion
        curr_price = safe_get_numeric(row, 'curr_price') or safe_get_numeric(row, 'current_price', 0)
        strike_price = safe_get_numeric(row, 'strike_price', 0)
        
        # Calculate moneyness - for calls: OTM when strike > current, for puts: OTM when strike < current
        if is_put:
            # For puts: OTM when strike < current price, ITM when strike > current price
            moneyness = ((curr_price - strike_price) / curr_price * 100) if curr_price != 0 and strike_price != 0 else 0
            moneyness_label = "OTM" if moneyness > 0 else ("ITM" if moneyness < 0 else "ATM")
        else:
            # For calls: OTM when strike > current price, ITM when strike < current price
            moneyness = ((strike_price - curr_price) / curr_price * 100) if curr_price != 0 and strike_price != 0 else 0
            moneyness_label = "OTM" if moneyness > 0 else ("ITM" if moneyness < 0 else "ATM")
        
        l_strike = safe_get_numeric(row, 'l_strike', 0)
        spread_width = l_strike - strike_price if l_strike != 0 and strike_price != 0 else 0
        
        l_delta = safe_get_numeric(row, 'l_delta', 0)
        delta = safe_get_numeric(row, 'delta', 0)
        delta_diff = l_delta - delta
        
        net_premium = safe_get_numeric(row, 'net_premium', 0)
        roi = (net_premium / 100000 * 100) if net_premium != 0 else 0
        
        # Calculate score
        score = 0
        # Calculate score using safe numeric conversion
        net_daily_val = safe_get_numeric(row, 'net_daily_premi', 0)
        if net_daily_val > 10000: score += 3
        elif net_daily_val > 7000: score += 2
        elif net_daily_val > 0: score += 1
        
        volume = safe_get_numeric(row, 'volume', 0)
        if volume > 1000: score += 3
        elif volume > 300: score += 2
        elif volume > 100: score += 1
        
        delta_score = safe_get_numeric(row, 'delta', 0)
        # For puts, delta is negative, so use absolute value
        delta_abs = abs(delta_score) if delta_score else 0
        if delta_abs > 0:
            if delta_abs < 0.35: score += 3
            elif delta_abs < 0.50: score += 2
            else: score += 1
        
        pe_ratio_score = safe_get_numeric(row, 'pe_ratio', 0)
        if pe_ratio_score > 0:
            if pe_ratio_score < 25: score += 2
            elif pe_ratio_score < 50: score += 1
        
        # Determine recommendation
        if score >= 9:
            recommendation = "STRONG BUY - Excellent risk/reward"
        elif score >= 7:
            recommendation = "BUY - Good opportunity"
        elif score >= 5:
            recommendation = "HOLD - Acceptable but monitor"
        else:
            recommendation = "PASS - Better opportunities available"
        
        # Assignment risk - use the delta_score we already calculated
        # For puts, delta is negative, so we need to check absolute value
        delta_abs = abs(delta_score) if delta_score else 0
        if delta_abs > 0:
            if delta_abs < 0.35:
                if is_put:
                    assignment_risk = "LOW - Strike is well OTM (below current price)"
                else:
                    assignment_risk = "LOW - Strike is well OTM (above current price)"
                risk_class = "risk-low"
            elif delta_abs < 0.50:
                assignment_risk = "MODERATE - Near ATM, watch closely"
                risk_class = "risk-moderate"
            else:
                if is_put:
                    assignment_risk = "HIGH - ITM or very close (strike above current), likely assignment"
                else:
                    assignment_risk = "HIGH - ITM or very close (strike below current), likely assignment"
                risk_class = "risk-high"
        else:
            assignment_risk = "UNKNOWN"
            risk_class = "risk-moderate"
        
        # Liquidity - use the volume we already calculated
        if volume > 0:
            if volume > 1000:
                liquidity = "EXCELLENT - Very liquid"
            elif volume > 300:
                liquidity = "GOOD - Adequate liquidity"
            elif volume > 100:
                liquidity = "FAIR - May have wider spreads"
            else:
                liquidity = "POOR - Low liquidity, watch bid-ask"
        else:
            liquidity = "UNKNOWN"
        
        # Valuation - use the pe_ratio_score we already calculated
        if pe_ratio_score > 0:
            if pe_ratio_score < 15:
                valuation = "ATTRACTIVE - Trading at discount"
            elif pe_ratio_score < 25:
                valuation = "FAIR - Reasonably valued"
            elif pe_ratio_score < 50:
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
        # Handle both curr_price and current_price column names
        curr_price_val = safe_get_numeric(row, 'curr_price', 0) or safe_get_numeric(row, 'current_price', 0)
        option_type_display = option_type.upper() if option_type else "CALL"
        html_parts.append(f'                    <p><span class="label">Option Type:</span> {option_type_display}</p>\n')
        html_parts.append(f'                    <p><span class="label">Current Price:</span> ${curr_price_val:.2f}</p>\n')
        html_parts.append(f'                    <p><span class="label">Short Strike:</span> ${strike_price:.2f} ({abs(moneyness):.2f}% {moneyness_label})</p>\n')
        html_parts.append(f'                    <p><span class="label">Long Strike:</span> ${l_strike:.2f}</p>\n')
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
        # Safely get numeric values for display
        days_to_expiry = safe_get_numeric(row, 'days_to_expiry', 0)
        opt_prem = safe_get_numeric(row, 'opt_prem.', 0)
        s_prem_tot = safe_get_numeric(row, 's_prem_tot', 0)
        num_contracts_display = safe_get_numeric(row, 'num_contracts', 0)
        l_days_to_expiry = safe_get_numeric(row, 'l_days_to_expiry', 0)
        l_prem = safe_get_numeric(row, 'l_prem', 0) or safe_get_numeric(row, 'l_opt_prem', 0)
        buy_cost_val = safe_get_numeric(row, "buy_cost", 0) or safe_get_numeric(row, "l_prem_tot", 0)
        
        html_parts.append(f'                        <p>│  Strike: ${strike_price:.2f} | Expiry: {html.escape(exp_date)} ({int(days_to_expiry)} DTE)</p>\n')
        html_parts.append(f'                        <p>│  Premium: ${opt_prem:.2f} per contract | Bid:Ask: {bid_ask_short}</p>\n')
        html_parts.append(f'                        <p>│  Total Credit: ${s_prem_tot:,.0f} ({int(num_contracts_display)} contracts)</p>\n')
        html_parts.append('                        <p>│</p>\n')
        html_parts.append(f'                        <p><span class="long">└─ LONG (BUY):</span> {option_ticker_long}</p>\n')
        html_parts.append(f'                        <p>   Strike: ${l_strike:.2f} | Expiry: {html.escape(l_exp_date)} ({int(l_days_to_expiry)} DTE)</p>\n')
        html_parts.append(f'                        <p>   Premium: ${l_prem:.2f} per contract | Bid:Ask: {bid_ask_long}</p>\n')
        html_parts.append(f'                        <p>   Total Debit: ${buy_cost_val:,.0f} ({int(num_contracts_display)} contracts)</p>\n')
        html_parts.append('                    </div>\n')
        html_parts.append('                </div>\n')
        
        # Premium Breakdown
        html_parts.append('                <div class="analysis-section">\n')
        html_parts.append('                    <h4>💰 PREMIUM BREAKDOWN</h4>\n')
        # Safely get premium values
        s_prem_tot_display = safe_get_numeric(row, "s_prem_tot", 0)
        s_day_prem = safe_get_numeric(row, "s_day_prem", 0)
        html_parts.append(f'                    <p><span class="label">Short Premium:</span> ${s_prem_tot_display:,.0f} (${s_day_prem:,.0f}/day)</p>\n')
        html_parts.append(f'                    <p><span class="label">Long Premium:</span> ${buy_cost_val:,.0f}</p>\n')
        html_parts.append(f'                    <p><span class="label">Net Credit:</span> ${net_premium:,.0f}</p>\n')
        # Safely format net_daily_premi
        net_daily_val_display = safe_get_numeric(row, "net_daily_premi", 0)
        html_parts.append(f'                    <p><span class="label">Daily Income:</span> ${net_daily_val_display:,.2f}</p>\n')
        html_parts.append(f'                    <p><span class="label">ROI on $100k:</span> {roi:.2f}%</p>\n')
        html_parts.append('                </div>\n')
        
        # Spread & Liquidity Analysis (if available)
        if 'spread_impact_pct' in row and pd.notna(row.get('spread_impact_pct')):
            spread_slippage = safe_get_numeric(row, 'spread_slippage', 0)
            net_after_spread = safe_get_numeric(row, 'net_premium_after_spread', net_premium)
            # Safely get net_daily_premi value
            net_daily_fallback = safe_get_numeric(row, 'net_daily_premi', 0)
            net_daily_after = safe_get_numeric(row, 'net_daily_premium_after_spread', net_daily_fallback)
            spread_impact = safe_get_numeric(row, 'spread_impact_pct', 0)
            liquidity_score = safe_get_numeric(row, 'liquidity_score', 0)
            assignment_risk = safe_get_numeric(row, 'assignment_risk', 0)
            trade_quality = safe_get_numeric(row, 'trade_quality', 0)
            
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
        # For puts, delta is negative, but display absolute value with note
        delta_display = delta if not is_put or delta >= 0 else f"{delta:.2f} (put delta)"
        l_delta_display = l_delta if not is_put or l_delta >= 0 else f"{l_delta:.2f} (put delta)"
        html_parts.append(f'                    <p><span class="label">Short Delta:</span> {delta_display} | <span class="label">Long Delta:</span> {l_delta_display} | <span class="label">Net:</span> {delta_diff:.3f}</p>\n')
        # Safely get theta values
        theta = safe_get_numeric(row, 'theta', 0)
        l_theta = safe_get_numeric(row, 'l_theta', 0)
        html_parts.append(f'                    <p><span class="label">Short Theta:</span> {theta:.2f} | <span class="label">Long Theta:</span> {l_theta:.2f}</p>\n')
        html_parts.append('                </div>\n')
        
        # Liquidity & Fundamentals
        html_parts.append('                <div class="analysis-section">\n')
        html_parts.append('                    <h4>🔄 LIQUIDITY & FUNDAMENTALS</h4>\n')
        # Safely format numeric values using helper function
        volume_display = volume if volume > 0 else 0
        num_contracts = safe_get_numeric(row, 'num_contracts', 0)
        pe_ratio_display = pe_ratio_score if pe_ratio_score > 0 else 0
        market_cap = safe_get_numeric(row, 'market_cap_b', 0)
        
        html_parts.append(f'                    <p><span class="label">Volume:</span> {volume_display:,.0f} contracts</p>\n')
        html_parts.append(f'                    <p><span class="label">Num Contracts:</span> {num_contracts:.0f}</p>\n')
        html_parts.append(f'                    <p><span class="label">P/E Ratio:</span> {pe_ratio_display:.2f}</p>\n')
        html_parts.append(f'                    <p><span class="label">Market Cap:</span> ${market_cap:.2f}B</p>\n')
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


def _prepare_dataframe_for_display(df: pd.DataFrame) -> tuple:
    """Prepare DataFrame for display by formatting columns and renaming.
    
    Args:
        df: Raw DataFrame to prepare
        
    Returns:
        Tuple of (df_display, df_raw) where df_display is formatted and df_raw is original
    """
    # Prepare data for HTML table
    df_display = df.copy()
    
    # Store raw values before formatting (for filtering) - keep original DataFrame
    df_raw = df.copy()
    
    # Format numeric columns for better display
    # Note: expiration_date columns are excluded - they will be formatted as dates separately
    numeric_cols = [
        'ticker','pe_ratio','market_cap_b','curr_price','strike_price','price_above_curr','opt_prem.','IV','delta','theta','days_to_expiry','s_prem_tot','s_day_prem','l_strike','l_prem','liv','l_delta','l_theta','l_days_to_expiry','l_prem_tot','l_cnt_avl','prem_diff','net_premium','net_daily_premi','volume','num_contracts','option_ticker','l_option_ticker',
        'spread_slippage','net_premium_after_spread','net_daily_premium_after_spread','spread_impact_pct','liquidity_score','assignment_risk','trade_quality'
    ]
    
    # Helper function to normalize column names
    def _normalize_col_name(name: str) -> str:
        return str(name).strip().lower().replace(' ', '_')
    
    # Also check for common column name variations
    all_numeric_cols = set(numeric_cols)
    for col in df_display.columns:
        col_lower = col.lower()
        # Map common variations
        if 'pe' in col_lower and 'ratio' in col_lower:
            all_numeric_cols.add(col)
        elif 'market' in col_lower and 'cap' in col_lower:
            all_numeric_cols.add(col)
        elif 'current' in col_lower or 'curr' in col_lower:
            all_numeric_cols.add(col)
        elif 'strike' in col_lower:
            all_numeric_cols.add(col)
        elif 'premium' in col_lower:
            all_numeric_cols.add(col)
        elif 'delta' in col_lower or 'theta' in col_lower:
            all_numeric_cols.add(col)
        elif 'volume' in col_lower or 'contracts' in col_lower:
            all_numeric_cols.add(col)
        elif 'days' in col_lower:
            all_numeric_cols.add(col)
    
    # Format expiration date columns to show only date portion (no time)
    def format_date_value(x, col_name):
        """Format date values to show only date portion."""
        if pd.isna(x) or x == '' or x is None:
            return ''
        normalized_col = _normalize_col_name(col_name)
        if normalized_col in ['expiration_date', 'l_expiration_date', 'long_expiration_date']:
            try:
                # First try to parse as datetime/timestamp
                if isinstance(x, pd.Timestamp):
                    return x.strftime('%Y-%m-%d')
                elif isinstance(x, (str, datetime)):
                    dt = pd.to_datetime(x, errors='coerce')
                    if pd.notna(dt):
                        return dt.strftime('%Y-%m-%d')
                
                # If that fails, try to parse as numeric (might be a year like 2025.00)
                x_str = str(x).strip()
                try:
                    # Try to parse as float first (handles cases like 2025.00)
                    num_val = float(x_str)
                    # If it's a reasonable year (1900-2100), it might be a year
                    if 1900 <= num_val <= 2100:
                        # Check if it's just a year (like 2025.00) - return as-is for now
                        # But try to parse the original value from df_raw if available
                        pass
                except (ValueError, TypeError):
                    pass
                
                # Try to extract date portion from string
                if ' ' in x_str:
                    return x_str.split(' ')[0]
                elif 'T' in x_str:
                    return x_str.split('T')[0]
                elif len(x_str) >= 10:
                    return x_str[:10]
                else:
                    # Return the full string if it's short enough
                    return x_str
            except (ValueError, TypeError, AttributeError):
                # If all parsing fails, return the string representation
                return str(x)
        elif normalized_col in ['latest_option_writets', 'latest_opt_ts']:
            # For latest_option_writets, it's stored as age in seconds (not a timestamp)
            # Format as human-readable age (secs, mins, hrs, days)
            try:
                # First check if it's already a timestamp (pd.Timestamp or datetime)
                if isinstance(x, pd.Timestamp):
                    # If it's a timestamp, calculate age from now
                    now = pd.Timestamp.now(tz='UTC') if x.tz else pd.Timestamp.now()
                    if x.tz:
                        age_sec = (now - x).total_seconds()
                    else:
                        age_sec = (now.tz_localize(None) - x).total_seconds()
                    return format_age_seconds(age_sec)
                elif isinstance(x, (str, datetime)):
                    # Try to parse as datetime first
                    dt = pd.to_datetime(x, errors='coerce')
                    if pd.notna(dt):
                        # It's a timestamp, calculate age
                        now = pd.Timestamp.now(tz='UTC') if dt.tz else pd.Timestamp.now()
                        if dt.tz:
                            age_sec = (now - dt).total_seconds()
                        else:
                            age_sec = (now.tz_localize(None) - dt).total_seconds()
                        return format_age_seconds(age_sec)
                
                # If not a timestamp, treat as age in seconds (numeric)
                x_str = str(x).strip()
                
                # Check if it looks like a timestamp string (has date-like patterns)
                if ' ' in x_str and len(x_str) >= 19:
                    # Looks like a timestamp string, try to parse
                    dt = pd.to_datetime(x_str, errors='coerce')
                    if pd.notna(dt):
                        now = pd.Timestamp.now(tz='UTC') if dt.tz else pd.Timestamp.now()
                        if dt.tz:
                            age_sec = (now - dt).total_seconds()
                        else:
                            age_sec = (now.tz_localize(None) - dt).total_seconds()
                        return format_age_seconds(age_sec)
                    # If parsing fails, return as-is
                    return x_str[:19]
                elif 'T' in x_str:
                    # ISO format timestamp
                    dt = pd.to_datetime(x_str, errors='coerce')
                    if pd.notna(dt):
                        now = pd.Timestamp.now(tz='UTC') if dt.tz else pd.Timestamp.now()
                        if dt.tz:
                            age_sec = (now - dt).total_seconds()
                        else:
                            age_sec = (now.tz_localize(None) - dt).total_seconds()
                        return format_age_seconds(age_sec)
                    return x_str.replace('T', ' ')[:19]
                else:
                    # Likely a numeric value (age in seconds)
                    # Try to parse as float
                    try:
                        age_sec = float(x_str)
                        # If it's a reasonable age value (0 to 1 year in seconds)
                        if 0 <= age_sec <= 31536000:  # 1 year in seconds
                            return format_age_seconds(age_sec)
                        # If it's a very large number, might be a timestamp in milliseconds
                        elif age_sec > 1000000000000:  # Timestamp in milliseconds
                            dt = pd.to_datetime(age_sec / 1000, unit='s', errors='coerce')
                            if pd.notna(dt):
                                now = pd.Timestamp.now(tz='UTC')
                                age_sec_calc = (now - dt.tz_localize('UTC')).total_seconds()
                                return format_age_seconds(age_sec_calc)
                    except (ValueError, TypeError):
                        pass
                    
                    # If all else fails, return as string
                    return x_str
            except (ValueError, TypeError, AttributeError):
                # If all parsing fails, return the string representation
                return str(x)
        return x
    
    # Format date columns first - use raw values from df_raw before any numeric formatting
    for col in df_display.columns:
        normalized_col = _normalize_col_name(col)
        if normalized_col in ['expiration_date', 'l_expiration_date', 'long_expiration_date', 'latest_option_writets', 'latest_opt_ts']:
            # Use raw values from df_raw to get original date format before any numeric conversion
            if df_raw is not None and col in df_raw.columns:
                df_display[col] = df_raw[col].apply(lambda x: format_date_value(x, col))
            else:
                df_display[col] = df_display[col].apply(lambda x: format_date_value(x, col))
            # Remove from numeric columns so they don't get formatted as numbers
            all_numeric_cols.discard(col)
    
    for col in all_numeric_cols:
        if col in df_display.columns:
            df_display[col] = df_display[col].apply(lambda x: format_numeric_value(x, col))
    
    # Replace remaining NaN with empty strings for display
    df_display = df_display.fillna('')
    
    # Helper function to find column by partial name match (handles truncated variations)
    def find_column_by_partial_name(df, target_pattern):
        """Find a column in DataFrame by partial name match, handling variations."""
        target_pattern_normalized = target_pattern.lower().replace(' ', '_')
        for col in df.columns:
            col_normalized = _normalize_col_name(col)
            # Check if the column starts with the pattern (handles truncation)
            if col_normalized.startswith(target_pattern_normalized):
                return col
            # Also check exact match
            if col_normalized == target_pattern_normalized:
                return col
        return None
    
    # Normalize/standardize column names first
    # Handle current_price variations (curr_price, current_price, cur_price)
    current_price_col = find_column_by_partial_name(df_display, 'curr')
    if current_price_col:
        # Check if it's actually a price column (not current_strike, etc.)
        if 'price' in current_price_col.lower() or current_price_col.lower() in ['curr_price', 'cur_price']:
            if current_price_col != 'current_price':
                df_display = df_display.rename(columns={current_price_col: 'current_price'})
                df_raw = df_raw.rename(columns={current_price_col: 'current_price'})
    
    # Handle price_with_change variations (price_with_chan, price_with_ch, price_with_change)
    price_with_change_col = find_column_by_partial_name(df_display, 'price_with_ch')
    if price_with_change_col and 'pct' not in price_with_change_col.lower():
        if price_with_change_col != 'price_with_change':
            df_display = df_display.rename(columns={price_with_change_col: 'price_with_change'})
            df_raw = df_raw.rename(columns={price_with_change_col: 'price_with_change'})
    
    # Handle price_change_pct variations (price_change_pc, price_change_pct)
    for col in df_display.columns:
        col_lower = col.lower()
        if 'price_change' in col_lower and 'pc' in col_lower and col != 'price_change_pct':
            df_display = df_display.rename(columns={col: 'price_change_pct'})
            df_raw = df_raw.rename(columns={col: 'price_change_pct'})
            break
    
    # Rename latest_opt_ts to latest_option_writets for display (before reordering)
    if 'latest_opt_ts' in df_display.columns:
        df_display = df_display.rename(columns={'latest_opt_ts': 'latest_option_writets'})
        if 'latest_opt_ts' in df_raw.columns:
            df_raw = df_raw.rename(columns={'latest_opt_ts': 'latest_option_writets'})
    
    # Reorder columns: ticker, pe_ratio, market_cap_b, current_price, price_with_change (renamed to change_pct)
    # Define desired column order (only reorder if columns exist)
    # Note: price_change_pct is kept for sorting but not shown in desired_order
    
    # Build a flexible mapping for column ordering that handles CSV column name variations
    def find_matching_column(desired_name):
        """Find a column that matches the desired name, handling variations."""
        # Exact match first
        if desired_name in df_display.columns:
            return desired_name
        
        # Handle common variations
        variations = {
            'current_price': ['curr_price', 'cur_price', 'current_price'],
            'price_with_change': ['price_with_change', 'price_with_chan', 'price_with_ch'],
            'option_premium': ['opt_prem.', 'opt_prem', 'option_premium'],
            'bid_ask': ['bid:ask', 'bid_ask'],
            'short_premium_total': ['s_prem_tot', 's_prem_total', 'short_premium_total'],
            'short_daily_premium': ['s_day_prem', 's_daily_prem', 'short_daily_premium'],
            'long_strike_price': ['l_strike', 'l_strike_price', 'long_strike_price'],
            'long_option_premium': ['l_opt_prem', 'l_prem', 'l_option_premium', 'long_option_premium'],
            'long_bid_ask': ['l_bid:ask', 'l_bid_ask', 'long_bid_ask'],
            'long_implied_volatility': ['liv', 'l_iv', 'long_implied_volatility'],
            'long_delta': ['l_delta', 'long_delta'],
            'long_theta': ['l_theta', 'long_theta'],
            'long_expiration_date': ['l_expiration_date', 'long_expiration_date'],
            'long_days_to_expiry': ['l_days_to_expiry', 'long_days_to_expiry'],
            'long_premium_total': ['l_prem_tot', 'l_premium_total', 'long_premium_total'],
            'long_contracts_available': ['l_cnt_avl', 'l_contracts_available', 'long_contracts_available'],
            'net_daily_premium': ['net_daily_premi', 'net_daily_premium'],
        }
        
        if desired_name in variations:
            for var in variations[desired_name]:
                if var in df_display.columns:
                    return var
        
        return None
    
    # Define desired order with standardized names
    desired_order_names = [
        'ticker', 'option_type', 'pe_ratio', 'market_cap_b', 'current_price', 'price_with_change',
        'strike_price', 'option_premium', 'expiration_date', 'bid_ask', 'delta', 'theta',
        'short_premium_total', 'short_daily_premium',
        'long_strike_price', 'long_option_premium', 'long_expiration_date', 'long_bid_ask', 'long_delta', 'long_theta',
        'long_implied_volatility', 'long_days_to_expiry', 
        'long_premium_total', 'long_contracts_available',
        'net_premium', 'net_daily_premium',
        'price_above_current', 'premium_above_diff_percentage',
        'implied_volatility', 'days_to_expiry',
        'potential_premium', 'daily_premium',
        'volume', 'num_contracts', 'option_ticker', 'long_option_ticker',
        'premium_diff',  # Will be hidden by default
        'spread_slippage', 'net_premium_after_spread', 'net_daily_premium_after_spread',
        'spread_impact_pct', 'liquidity_score', 'assignment_risk', 'trade_quality',
        'latest_option_writets'  # Latest option write timestamp (always visible, rightmost)
    ]
    
    # Get existing columns in desired order, handling variations
    ordered_cols = []
    for desired_name in desired_order_names:
        actual_col = find_matching_column(desired_name)
        if actual_col:
            ordered_cols.append(actual_col)
    
    # Remove latest_option_writets from ordered_cols if present (will be added at the end)
    # latest_option_writets should always be at the rightmost position
    if 'latest_option_writets' in ordered_cols:
        ordered_cols.remove('latest_option_writets')
    
    # Add any remaining columns (excluding price_change_pct and latest_option_writets from visible)
    remaining_cols = [col for col in df_display.columns 
                      if col not in ordered_cols 
                      and col != 'price_change_pct' 
                      and col != 'latest_option_writets']
    # Add price_change_pct before latest_option_writets (hidden, used for sorting)
    if 'price_change_pct' in df_display.columns:
        remaining_cols.append('price_change_pct')
    # Add latest_option_writets at the very end (rightmost position) if it exists
    if 'latest_option_writets' in df_display.columns:
        remaining_cols.append('latest_option_writets')
    df_display = df_display[ordered_cols + remaining_cols]
    
    # Rename price_with_change to change_pct for display
    if 'price_with_change' in df_display.columns:
        df_display = df_display.rename(columns={'price_with_change': 'change_pct'})
        # Also update df_raw for consistency
        if 'price_with_change' in df_raw.columns:
            df_raw = df_raw.rename(columns={'price_with_change': 'change_pct'})
    
    # Rename l_prem to l_opt_prem for display
    if 'l_prem' in df_display.columns:
        df_display = df_display.rename(columns={'l_prem': 'l_opt_prem'})
        if 'l_prem' in df_raw.columns:
            df_raw = df_raw.rename(columns={'l_prem': 'l_opt_prem'})
    
    # Rename l_prem_tot to buy_cost for display
    if 'l_prem_tot' in df_display.columns:
        df_display = df_display.rename(columns={'l_prem_tot': 'buy_cost'})
        if 'l_prem_tot' in df_raw.columns:
            df_raw = df_raw.rename(columns={'l_prem_tot': 'buy_cost'})
    
    return df_display, df_raw


def generate_html_output(df: pd.DataFrame, output_dir: str) -> None:
    """Generate HTML output with sortable table.
    
    Args:
        df: DataFrame with the results
        output_dir: Directory path where to create the HTML output
    """
    # Use new modular implementation if available
    if _USE_NEW_MODULES:
        return _generate_html_output(df, output_dir)
    
    # Fallback to original implementation below
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Helper function to normalize column names (used in helper functions)
    def _normalize_col_name(name: str) -> str:
        return str(name).strip().lower().replace(' ', '_')
    
    # Split data into calls and puts if option_type column exists
    df_calls = pd.DataFrame()
    df_puts = pd.DataFrame()
    has_calls = False
    has_puts = False
    
    if 'option_type' in df.columns:
        df_calls = df[df['option_type'].str.lower() == 'call'].copy() if 'option_type' in df.columns else pd.DataFrame()
        df_puts = df[df['option_type'].str.lower() == 'put'].copy() if 'option_type' in df.columns else pd.DataFrame()
        has_calls = len(df_calls) > 0
        has_puts = len(df_puts) > 0
    else:
        # If no option_type column, treat all as calls (backward compatibility)
        df_calls = df.copy()
        has_calls = True
    
    # Prepare DataFrames for display
    df_calls_display, df_calls_raw = _prepare_dataframe_for_display(df_calls) if has_calls else (pd.DataFrame(), pd.DataFrame())
    df_puts_display, df_puts_raw = _prepare_dataframe_for_display(df_puts) if has_puts else (pd.DataFrame(), pd.DataFrame())
    df_display, df_raw = _prepare_dataframe_for_display(df)  # For comprehensive analysis
    
    # Get current timestamp for display
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
    # Get ISO timestamp with timezone for JavaScript parsing (ensure it's parseable)
    # Format as ISO string with 'Z' suffix for UTC to ensure JavaScript can parse it
    if now.tzinfo is None:
        # If no timezone info, format as UTC
        iso_timestamp = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    else:
        iso_timestamp = now.isoformat()
    
    # Generate HTML - build it piece by piece
    html_parts = []
    
    # HTML head and styles
    html_parts.append("""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Options Analysis Results</title>
    <style>
""")
    html_parts.append(get_css_styles())
    html_parts.append("""    </style>
</head>
<body>
    <div class="container">
        <div class="header">
""")
    
    # Detect option type from data for title
    option_type_detected = 'call'  # Default
    if has_calls and has_puts:
        option_type_detected = 'mixed'
    elif has_puts:
        option_type_detected = 'put'
    
    # Set title based on option type
    if option_type_detected == 'put':
        title = "📊 Cash-Secured Puts Analysis Results"
    elif option_type_detected == 'mixed':
        title = "📊 Options Analysis Results (Calls & Puts)"
    else:
        title = "📊 Covered Calls Analysis Results"
    
    html_parts.append(f'            <h1>{title}</h1>\n')
    html_parts.append("""            <div class="tabs">\n""")
    
    # Dynamically create tab buttons based on available data
    tab_index = 0
    if has_calls:
        html_parts.append(f'                <button class="tab-button {"active" if tab_index == 0 else ""}" onclick="switchTab({tab_index})">📞 Calls</button>\n')
        tab_index += 1
    if has_puts:
        html_parts.append(f'                <button class="tab-button {"active" if tab_index == 0 else ""}" onclick="switchTab({tab_index})">📉 Puts</button>\n')
        tab_index += 1
    # Always add comprehensive analysis tab
    html_parts.append(f'                <button class="tab-button" onclick="switchTab({tab_index})">📊 Comprehensive Analysis</button>\n')
    html_parts.append("""            </div>
            <p id="generatedTime" data-generated="""" + iso_timestamp + """">Data updated: <span id="dataTimestamp">""" + timestamp + """</span> <span id="timeAgo"></span></p>
            <p class="desktop-only">Click column headers to sort • """ + str(len(df)) + """ total results</p>
            <p class="mobile-only">Tap cards to expand details • """ + str(len(df)) + """ total results</p>
        </div>
""")
    
    # Generate tabs for calls and puts
    tab_index = 0
    if has_calls:
        html_parts.append(_generate_table_and_cards_html(df_calls_display, df_calls_raw, 'calls', tab_index == 0, _normalize_col_name))
        tab_index += 1
    if has_puts:
        html_parts.append(_generate_table_and_cards_html(df_puts_display, df_puts_raw, 'puts', tab_index == 0, _normalize_col_name))
        tab_index += 1
    
    # Comprehensive analysis tab
    html_parts.append(f"""        <div class="tab-content">
""")
    html_parts.append(generate_detailed_analysis_html(df_display))
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


def _generate_table_and_cards_html(df_display: pd.DataFrame, df_raw: pd.DataFrame, prefix: str, is_active: bool, _normalize_col_name) -> str:
    """Generate HTML for table, filters, and cards for a given DataFrame and prefix.
    
    Args:
        df_display: Formatted DataFrame for display
        df_raw: Raw DataFrame with original values for filtering
        prefix: Prefix for IDs ('calls' or 'puts')
        is_active: Whether this tab should be active by default
        _normalize_col_name: Function to normalize column names
        
    Returns:
        String containing HTML for the tab content
    """
    if df_display.empty:
        return f"""        <div class="tab-content{' active' if is_active else ''}">
            <div style="padding: 20px; text-align: center;">
                <p>No {prefix} data available.</p>
            </div>
        </div>
"""
    
    html_parts = []
    active_class = ' active' if is_active else ''
    html_parts.append(f"""        <div class="tab-content{active_class}">
        <div style="margin-bottom: 15px; display: flex; justify-content: space-between; gap: 10px; align-items: center;">
            <div>
                <button class="filter-button clear" onclick="toggleHiddenColumns('{prefix}')" id="{prefix}toggleHiddenBtn" title="Show or hide default-hidden columns">
                    👁️ Show hidden columns
                </button>
            </div>
            <div style="text-align: right;">
                <button class="filter-button" onclick="toggleFilterSection('{prefix}')" id="{prefix}toggleFilterBtn" title="Show/hide filter options and filterable column names">
                    🔍 Filter
                </button>
            </div>
        </div>
        <div class="filter-section" id="{prefix}filterSection">
            <h3 style="margin-top: 0; color: #667eea;">🔍 Filter Options</h3>
            <div class="filter-logic">
                <label>Filter Logic:</label>
                <label><input type="radio" name="{prefix}filterLogic" value="AND" checked onchange="updateFilterLogic('{prefix}', 'AND')"> AND</label>
                <label><input type="radio" name="{prefix}filterLogic" value="OR" onchange="updateFilterLogic('{prefix}', 'OR')"> OR</label>
            </div>
            <div class="filter-controls">
                <div class="filter-input-group">
                    <input type="text" id="{prefix}filterInput" class="filter-input" placeholder="e.g., pe_ratio > 20, volume exists, net_daily_premium > 100" onkeypress="handleFilterKeyPress(event, '{prefix}')">
                    <button class="filter-button" onclick="addFilter('{prefix}')">Add Filter</button>
                    <button class="filter-button clear" onclick="clearFilters('{prefix}')">Clear All</button>
                </div>
            </div>
            <div id="{prefix}filterError" class="filter-error"></div>
            <div id="{prefix}activeFilters" style="margin-top: 10px;"></div>
            <div class="filter-help">
                <strong>Filter Examples:</strong><br>
                • <code>pe_ratio > 20</code> - P/E ratio greater than 20<br>
                • <code>market_cap_b < 3.5</code> - Market cap less than 3.5B<br>
                • <code>volume exists</code> - Volume data exists<br>
                • <code>net_daily_premium > 100</code> - Net daily premium greater than 100<br>
                • <code>delta < 0.5</code> - Delta less than 0.5<br>
                • <code>days_to_expiry >= 7</code> - Days to expiry at least 7<br>
                • <code>num_contracts > volume</code> - Field-to-field comparison<br>
                • <code>curr_price*1.05 < strike_price</code> - Mathematical expression (5% above current price less than strike)<br>
                • <code>strike_price*0.95 > curr_price</code> - Mathematical expression (strike 5% below current)<br>
                <strong>Operators:</strong> <code>&gt;</code> <code>&gt;=</code> <code>&lt;</code> <code>&lt;=</code> <code>==</code> <code>!=</code> <code>exists</code> <code>not_exists</code><br>
                <strong>Math Operations:</strong> Use <code>+</code> <code>-</code> <code>*</code> <code>/</code> in expressions (e.g., <code>field*1.05</code>, <code>field+100</code>)<br>
                <strong>💡 Tip:</strong> When the filter section is expanded, column headers show their filterable field names. Filters are automatically saved in the URL - share the URL to share your filtered view!
            </div>
        </div>
        <div class="table-wrapper hide-hidden" id="{prefix}tableWrapper">
            <table id="{prefix}resultsTable">
                <thead>
                    <tr>
""")
    
    # Columns to hide by default (use normalized lowercase names with underscores)
    # Note: This is checked AFTER column renaming, so use the final column names
    hidden_columns_list = [
        'price_change_pct',  # Hidden - used only for sorting change% column
        'price_change_pc',   # Variation that might exist
        'price_above_curr',
        'price_above_current',
        'premium_diff',  # Hidden by default, show only with hidden columns
        'prem_diff',     # Variation
        'days_to_expiry',
        'iv',
        'implied_volatility',
        'liv',
        'long_implied_volatility',
        'l_days_to_expiry',
        'long_days_to_expiry',
        'long_contracts_available',
        'option_ticker',
        'l_option_ticker',
        'buy_cost',
        'net_premium_after_spread',
        'spread_slippage',
        'net_daily_premium_after_spread',
        'spread_impact_pct',
        'liquidity_score',
        'assignment_risk',
    ]

    hidden_columns_set = set(_normalize_col_name(col) for col in hidden_columns_list)
    
    # Columns to always hide (hidden in all cases, even when "Show hidden columns" is clicked)
    always_hidden_columns_list = [
        'l_cnt_avl',
        'long_contracts_available',  # Also hide the full name variation
    ]
    
    always_hidden_columns_set = set(_normalize_col_name(col) for col in always_hidden_columns_list)

    # Define column groups - pairs that should be grouped together
    # Format: (group_name, ([col1_variations], [col2_variations]))
    column_groups_def = {
        'strike_price': (['strike_price'], ['l_strike']),
        'opt_prem': (['opt_prem.', 'opt_prem', 'option_premium'], ['l_opt_prem', 'l_prem', 'long_option_premium']),
        'expiration_date': (['expiration_date'], ['l_expiration_date', 'long_expiration_date']),
        'bid:ask': (['bid:ask', 'bid_ask'], ['l_bid:ask', 'l_bid_ask', 'long_bid_ask']),
        'delta': (['delta'], ['l_delta', 'long_delta']),
        'theta': (['theta'], ['l_theta', 'long_theta']),
        's_prem_tot': (['s_prem_tot', 'short_premium_total'], ['net_premium']),
        's_day_prem': (['s_day_prem', 'short_daily_premium'], ['net_daily_premium', 'net_daily_premi']),
    }
    
    # Find actual column names in df_display that match the group definitions
    def find_matching_col(variations):
        """Find the first column in df_display that matches any of the variations."""
        for col in df_display.columns:
            if col in variations:
                return col
        return None
    
    # Build actual column groups with real column names
    column_groups = {}
    col_to_group = {}
    group_names = {}
    for group_name, (col1_variations, col2_variations) in column_groups_def.items():
        col1 = find_matching_col(col1_variations)
        col2 = find_matching_col(col2_variations)
        if col1 and col2:
            column_groups[group_name] = (col1, col2)
            col_to_group[col1] = group_name
            col_to_group[col2] = group_name
            # Use readable group names
            group_names[group_name] = {
                'strike_price': 'Strike Price',
                'opt_prem': 'Option Premium',
                'expiration_date': 'Expiration Date',
                'bid:ask': 'Bid:Ask',
                'delta': 'Delta',
                'theta': 'Theta',
                's_prem_tot': 'Premium Total',
                's_day_prem': 'Daily Premium',
            }.get(group_name, group_name.replace('_', ' ').title())
    
    # Reorder columns so grouped pairs are adjacent
    # Strategy: 
    # 1. Keep ungrouped columns in their original order (at the beginning), except for specific ones
    # 2. Place grouped pairs together (col1 immediately followed by col2)
    # 3. Place num_contracts, buy_cost, volume, trade_quality after daily premium columns
    original_columns = list(df_display.columns)
    new_column_order = []
    processed_cols_reorder = set()
    
    # Columns to place after daily premium
    after_daily_premium = ['num_contracts', 'buy_cost', 'volume', 'trade_quality']
    
    # First, add ungrouped columns in their original order (excluding those that go after daily premium, price_change_pct, and latest_option_writets)
    # latest_option_writets should always be at the rightmost position (after price_change_pct)
    for col in original_columns:
        if col not in col_to_group and col not in after_daily_premium and col != 'latest_option_writets' and col != 'price_change_pct':
            new_column_order.append(col)
            processed_cols_reorder.add(col)
    
    # Then, add grouped pairs together
    # Use the order defined in column_groups_def to maintain a consistent order
    # expiration_date comes right after strike_price
    group_order = ['strike_price', 'expiration_date', 'opt_prem', 'bid:ask', 'delta', 'theta', 's_prem_tot', 's_day_prem']
    for group_name in group_order:
        if group_name in column_groups:
            col1, col2 = column_groups[group_name]
            if col1 not in processed_cols_reorder and col1 in original_columns:
                new_column_order.append(col1)
                processed_cols_reorder.add(col1)
            if col2 not in processed_cols_reorder and col2 in original_columns:
                new_column_order.append(col2)
                processed_cols_reorder.add(col2)
    
    # Add columns that go after daily premium
    for col in after_daily_premium:
        if col not in processed_cols_reorder and col in original_columns:
            new_column_order.append(col)
            processed_cols_reorder.add(col)
    
    # Add any remaining columns that weren't processed (shouldn't happen, but safety check)
    # But ensure price_change_pct and latest_option_writets are always at the rightmost positions
    latest_opt_col_reorder = None
    price_change_pct_col = None
    for col in original_columns:
        if col not in processed_cols_reorder:
            if col == 'latest_option_writets':
                latest_opt_col_reorder = col
            elif col == 'price_change_pct':
                price_change_pct_col = col
            else:
                new_column_order.append(col)
                processed_cols_reorder.add(col)
    
    # Remove latest_option_writets and price_change_pct from new_column_order if they were somehow added earlier
    if 'latest_option_writets' in new_column_order:
        new_column_order.remove('latest_option_writets')
        processed_cols_reorder.discard('latest_option_writets')
        latest_opt_col_reorder = 'latest_option_writets'
    if 'price_change_pct' in new_column_order:
        new_column_order.remove('price_change_pct')
        processed_cols_reorder.discard('price_change_pct')
        price_change_pct_col = 'price_change_pct'
    
    # Add price_change_pct before latest_option_writets (hidden, used for sorting)
    if price_change_pct_col or 'price_change_pct' in original_columns:
        if 'price_change_pct' in original_columns:
            new_column_order.append('price_change_pct')
            processed_cols_reorder.add('price_change_pct')
    
    # Add latest_option_writets at the very end (rightmost position) if it exists
    if latest_opt_col_reorder or 'latest_option_writets' in original_columns:
        if 'latest_option_writets' in original_columns:
            new_column_order.append('latest_option_writets')
            processed_cols_reorder.add('latest_option_writets')
    
    # Reorder both df_display and df_raw to match the new column order
    df_display = df_display[new_column_order]
    if df_raw is not None:
        # Only reorder columns that exist in df_raw
        df_raw_cols = [col for col in new_column_order if col in df_raw.columns]
        df_raw = df_raw[df_raw_cols]
    
    # Track which columns have been processed for the first row
    processed_cols_first_row = set()
    
    # Generate two-row header structure
    # First row: ONLY group headers (for grouped columns only)
    html_parts.append("""                    <tr class="group-header-row">\n""")
    for col in df_display.columns:
        if col in processed_cols_first_row:
            continue
        
        if col in col_to_group:
            # This column is part of a group
            group_name = col_to_group[col]
            col1, col2 = column_groups[group_name]
            
            # Check if both columns exist
            if col1 in df_display.columns and col2 in df_display.columns:
                # Only create group header if we encounter the first column of the pair
                if col == col1:
                    # Create group header spanning 2 columns
                    group_display_name = group_names[group_name]
                    html_parts.append(f'                        <th class="group-header" colspan="2">{html.escape(group_display_name)}</th>\n')
                    processed_cols_first_row.add(col1)
                    processed_cols_first_row.add(col2)
                # If col == col2, it will be skipped since it's already processed
            else:
                # Only one column of the group exists, don't create group header
                # Just mark it as processed so it doesn't get a header in first row
                processed_cols_first_row.add(col)
        else:
            # Regular column (not grouped) - create empty cell in first row
            normalized_col = _normalize_col_name(col)
            is_hidden = normalized_col in hidden_columns_set
            is_always_hidden = normalized_col in always_hidden_columns_set
            hidden_class = ' is-hidden-col' if is_hidden else ''
            always_hidden_class = ' always-hidden' if is_always_hidden else ''
            html_parts.append(f'                        <th class="group-header{hidden_class}{always_hidden_class}" colspan="1" style="background: transparent; border: none;"></th>\n')
            processed_cols_first_row.add(col)
    
    html_parts.append("""                    </tr>\n""")
    
    # Second row: individual column headers (sortable) for ALL columns
    # NOTE: Even though columns are visually grouped above, each column maintains its own
    # separate header, data cells, and can be sorted/filtered independently
    html_parts.append("""                    <tr class="column-header-row">\n""")
    for col in df_display.columns:
        col_index = df_display.columns.get_loc(col)
        truncated_title = truncate_header(str(col), 15)
        # Use the original column name as the filterable name
        filterable_name = html.escape(str(col))
        normalized_col = _normalize_col_name(col)
        is_hidden = normalized_col in hidden_columns_set
        is_always_hidden = normalized_col in always_hidden_columns_set
        hidden_class = ' is-hidden-col' if is_hidden else ''
        always_hidden_class = ' always-hidden' if is_always_hidden else ''
        
        # Determine if this column is part of a group and which position (short/long)
        grouped_class = ''
        short_long_label = ''
        if col in col_to_group:
            grouped_class = ' grouped-column'
            group_name = col_to_group[col]
            col1, col2 = column_groups[group_name]
            if col == col1:
                short_long_label = 'Short'
            elif col == col2:
                short_long_label = 'Long'
        
        html_parts.append(f'                        <th class="sortable{grouped_class}{hidden_class}{always_hidden_class}" onclick="sortTable(\'{prefix}\', {col_index})" data-filterable-name="{filterable_name}">')
        html_parts.append(f'                            <span class="column-name-display">{truncated_title}</span>')
        if short_long_label:
            html_parts.append(f'                            <span class="column-name-short-long">{short_long_label}</span>')
        html_parts.append(f'                            <span class="column-name-filterable">{filterable_name}</span>')
        html_parts.append(f'                        </th>\n')
    
    html_parts.append("""                    </tr>
                </thead>
                <tbody>
""")
    
    # Generate table rows with raw values stored in data attributes
    # NOTE: Each column maintains its own separate data cells, even when visually grouped
    for row_idx, row in df_display.iterrows():
        html_parts.append(f'                    <tr data-row-index="{row_idx}">\n')
        for col in df_display.columns:
            cell_value = str(row[col]) if pd.notna(row[col]) else ''
            # Escape HTML special characters
            cell_value = cell_value.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            
            # Get raw value for filtering
            raw_value = None
            raw_text = None
            
            # Normalize column name for use in multiple checks
            normalized_col = _normalize_col_name(col)
            
            # Check if this is a date column first (before numeric conversion)
            # Note: latest_option_writets is NOT a date column - it's age in seconds
            is_date_col = (
                normalized_col in ['expiration_date', 'l_expiration_date', 'long_expiration_date'] or
                normalized_col.endswith('_expiration_date') or
                normalized_col.endswith('_exp_date') or
                ('expiration_date' in normalized_col and 'latest_option_writets' not in normalized_col and 'latest_opt_ts' not in normalized_col)
            )
            
            # Check if this is latest_option_writets (age in seconds, not a timestamp)
            is_age_col = normalized_col in ['latest_option_writets', 'latest_opt_ts'] or normalized_col.endswith('_writets')
            
            if row_idx in df_raw.index and col in df_raw.columns:
                raw_val = df_raw.loc[row_idx, col]
                if pd.notna(raw_val):
                    # Handle latest_option_writets as age in seconds (numeric, not date)
                    if is_age_col:
                        try:
                            # Store as numeric seconds for sorting
                            val_str = str(raw_val).strip()
                            # Try direct conversion first
                            try:
                                age_sec = float(val_str)
                                # Store as numeric value for sorting (smaller = more recent)
                                raw_value = str(age_sec)
                            except (ValueError, TypeError):
                                # Extract first valid number from string
                                match = re.search(r'-?\d+\.?\d*', val_str)
                                if match:
                                    age_sec = float(match.group())
                                    raw_value = str(age_sec)
                                else:
                                    raw_text = str(raw_val)
                        except (ValueError, TypeError, AttributeError):
                            raw_text = str(raw_val)
                    # Handle date columns specially
                    elif is_date_col:
                        try:
                            # Try to parse as datetime/timestamp
                            if isinstance(raw_val, pd.Timestamp):
                                # Store as timestamp (milliseconds since epoch) for sorting
                                raw_value = str(int(raw_val.timestamp() * 1000))
                            elif isinstance(raw_val, (str, datetime)):
                                dt = pd.to_datetime(raw_val, errors='coerce')
                                if pd.notna(dt):
                                    raw_value = str(int(dt.timestamp() * 1000))
                            else:
                                # Try to parse string representation
                                val_str = str(raw_val).strip()
                                dt = pd.to_datetime(val_str, errors='coerce')
                                if pd.notna(dt):
                                    raw_value = str(int(dt.timestamp() * 1000))
                        except (ValueError, TypeError, AttributeError, OverflowError):
                            # If date parsing fails, fall back to text sorting
                            raw_text = str(raw_val)
                    else:
                        # Store numeric value if it's a number (non-date columns)
                        try:
                            # Handle malformed strings like '0.120.260.110.210.36'
                            val_str = str(raw_val)
                            # Try direct conversion first
                            try:
                                float_val = float(val_str)
                                # If successful and it's actually a number (not NaN), store as numeric
                                if not pd.isna(float_val):
                                    raw_value = str(float_val)
                                else:
                                    raw_text = str(raw_val)
                            except (ValueError, TypeError):
                                # Extract first valid number from string if direct conversion fails
                                match = re.search(r'-?\d+\.?\d*', val_str)
                                if match:
                                    try:
                                        float_val = float(match.group())
                                        if not pd.isna(float_val):
                                            raw_value = str(float_val)
                                        else:
                                            raw_text = str(raw_val)
                                    except (ValueError, TypeError):
                                        raw_text = str(raw_val)
                                else:
                                    # Not a number, store as text
                                    raw_text = str(raw_val)
                        except (ValueError, TypeError, AttributeError):
                            # Not a number, store as text
                            raw_text = str(raw_val)
            
            # For change_pct column, also store the percentage value from price_change_pct for sorting
            if col == 'change_pct' and 'price_change_pct' in df_raw.columns and row_idx in df_raw.index:
                pct_val = df_raw.loc[row_idx, 'price_change_pct']
                if pd.notna(pct_val):
                    try:
                        pct_float = float(pct_val)
                        if not pd.isna(pct_float):
                            # Store as raw value for sorting (this will be used by JavaScript)
                            raw_value = str(pct_float)
                    except (ValueError, TypeError):
                        pass
            
            # Calculate and append theta percentage for theta columns
            is_theta_col = (normalized_col == 'theta' or normalized_col == 'l_theta' or normalized_col == 'long_theta')
            if is_theta_col and row_idx in df_raw.index:
                # Get theta value from df_raw
                theta_val = None
                if col in df_raw.columns:
                    theta_raw = df_raw.loc[row_idx, col]
                    if pd.notna(theta_raw):
                        try:
                            theta_str = str(theta_raw)
                            # Try direct conversion
                            try:
                                theta_val = float(theta_str)
                            except (ValueError, TypeError):
                                # Extract first valid number
                                match = re.search(r'-?\d+\.?\d*', theta_str)
                                if match:
                                    theta_val = float(match.group())
                        except (ValueError, TypeError, AttributeError):
                            pass
                
                # Find corresponding option premium column
                opt_prem_col = None
                if normalized_col == 'theta':
                    # Short theta - find short option premium
                    for opt_col in ['opt_prem.', 'opt_prem', 'option_premium']:
                        if opt_col in df_raw.columns:
                            opt_prem_col = opt_col
                            break
                elif normalized_col in ['l_theta', 'long_theta']:
                    # Long theta - find long option premium
                    for opt_col in ['l_opt_prem', 'l_prem', 'long_option_premium']:
                        if opt_col in df_raw.columns:
                            opt_prem_col = opt_col
                            break
                
                # Calculate theta percentage
                if theta_val is not None and opt_prem_col and opt_prem_col in df_raw.columns:
                    opt_prem_raw = df_raw.loc[row_idx, opt_prem_col]
                    if pd.notna(opt_prem_raw):
                        try:
                            opt_prem_str = str(opt_prem_raw)
                            # Try direct conversion
                            try:
                                opt_prem_val = float(opt_prem_str)
                            except (ValueError, TypeError):
                                # Extract first valid number
                                match = re.search(r'-?\d+\.?\d*', opt_prem_str)
                                if match:
                                    opt_prem_val = float(match.group())
                                else:
                                    opt_prem_val = None
                            
                            if opt_prem_val is not None and opt_prem_val != 0:
                                theta_pct = (theta_val / opt_prem_val) * 100
                                # Validate theta percentage - if invalid (>100%, NaN, or infinity), show N/A
                                if pd.isna(theta_pct) or np.isinf(theta_pct) or abs(theta_pct) > 100:
                                    # Invalid percentage - show N/A
                                    if cell_value:
                                        cell_value = f"{cell_value} (N/A)"
                                    else:
                                        cell_value = f"{theta_val:.2f} (N/A)"
                                    # Don't set raw_value for invalid percentages - keep base theta value for sorting
                                else:
                                    # Valid percentage - store in raw_value for sorting
                                    raw_value = str(theta_pct)
                                    # Append theta percentage to cell value
                                    if cell_value:
                                        cell_value = f"{cell_value} ({theta_pct:.2f}%)"
                                    else:
                                        cell_value = f"{theta_val:.2f} ({theta_pct:.2f}%)"
                        except (ValueError, TypeError, AttributeError, ZeroDivisionError):
                            pass
            
            # Build td with data attributes
            td_attrs = []
            if raw_value is not None:
                td_attrs.append(f'data-raw="{html.escape(str(raw_value))}"')
            if raw_text is not None:
                td_attrs.append(f'data-raw-text="{html.escape(str(raw_text))}"')
            # Hidden class for default-hidden columns (normalized_col already calculated above)
            td_hidden_class = ' is-hidden-col' if normalized_col in hidden_columns_set else ''
            td_always_hidden_class = ' always-hidden' if normalized_col in always_hidden_columns_set else ''
            td_hidden_class = (td_hidden_class + td_always_hidden_class).strip()
            
            # Add price change color class for change_pct column (renamed from price_with_change)
            price_class = ''
            # Check if this is the change_pct column (handle both original and renamed names)
            is_change_col = (normalized_col == 'price_with_change' or 
                           normalized_col == 'change_pct' or
                           normalized_col == 'change%')
            if is_change_col and cell_value:
                # Check multiple format variations:
                # Format 1: "+$2.50 (+2.50%)" or "-$2.50 (-2.50%)"
                # Format 2: "$336.15 (-0.68%)" where sign is in the percentage
                # Format 3: "$2.50 (2.50%)" where positive doesn't have explicit sign
                if cell_value.startswith('+$') or (cell_value.startswith('$') and '(+' in cell_value):
                    price_class = ' price-positive'
                elif cell_value.startswith('-$') or (cell_value.startswith('$') and '(-' in cell_value):
                    price_class = ' price-negative'
                elif '(+' in cell_value:
                    price_class = ' price-positive'
                elif '(-' in cell_value:
                    price_class = ' price-negative'
            
            # Combine all classes
            all_classes = (td_hidden_class + price_class).strip()
            attrs_str = (' ' + ' '.join(td_attrs) if td_attrs else '')
            class_attr = f' class="{all_classes}"' if all_classes else ''
            html_parts.append(f'                        <td{class_attr}{attrs_str}>{cell_value}</td>\n')
        html_parts.append("                    </tr>\n")
    
    html_parts.append("""                </tbody>
            </table>
        </div>
        
        <!-- Mobile Card Layout -->
        <div class="card-wrapper" id="{prefix}cardWrapper">
""")
    
    # Define primary columns (always visible on cards) and expandable columns
    # Map normalized column names to display labels
    primary_columns_map = {
        'ticker': 'Ticker',
        'current_price': 'Price',
        'curr_price': 'Price',
        'change_pct': 'Change',
        'price_with_change': 'Change',
        'strike_price': 'Strike',
        'option_premium': 'Premium',
        'opt_prem': 'Premium',
        'expiration_date': 'Expiry',
        'daily_premium': 'Daily',
        'net_daily_premium': 'Net Daily',
        'net_daily_premi': 'Net Daily',
        's_day_prem': 'Daily',
        'short_daily_premium': 'Daily',
    }
    
    # Helper function to find column by normalized name
    def find_col_by_normalized(target_normalized):
        for col in df_display.columns:
            if _normalize_col_name(col) == target_normalized:
                return col
        return None
    
    # Generate cards for each row
    for row_idx, row in df_display.iterrows():
        # Get ticker for card header
        ticker = str(row.get('ticker', 'N/A')) if pd.notna(row.get('ticker')) else 'N/A'
        
        # Get current price and change - try multiple column name variations
        current_price_val = ''
        for price_col in ['current_price', 'curr_price', 'cur_price']:
            if price_col in row.index and pd.notna(row[price_col]) and str(row[price_col]).strip():
                current_price_val = row[price_col]
                break
        
        change_pct_val = ''
        for change_col in ['change_pct', 'price_with_change', 'price_with_chan']:
            if change_col in row.index and pd.notna(row[change_col]) and str(row[change_col]).strip():
                change_pct_val = row[change_col]
                break
        
        # Determine change color class
        change_class = ''
        if change_pct_val:
            change_str = str(change_pct_val)
            if '+$' in change_str or '(+' in change_str:
                change_class = 'positive'
            elif '-$' in change_str or '(-' in change_str:
                change_class = 'negative'
        
        # Store raw values for filtering - get from df_raw if available
        card_data_attrs = []
        if df_raw is not None and row_idx in df_raw.index:
            # Store key values as data attributes for filtering
            for col in ['ticker', 'current_price', 'strike_price', 'option_premium', 'daily_premium', 'net_daily_premium', 'volume', 'delta', 'theta']:
                col_found = None
                for c in df_raw.columns:
                    if _normalize_col_name(c) == _normalize_col_name(col):
                        col_found = c
                        break
                if col_found and col_found in df_raw.columns:
                    raw_val = df_raw.loc[row_idx, col_found]
                    if pd.notna(raw_val):
                        try:
                            # Try to get numeric value
                            if isinstance(raw_val, (int, float)):
                                card_data_attrs.append(f'data-{_normalize_col_name(col)}="{raw_val}"')
                            else:
                                val_str = str(raw_val)
                                # Try to extract number
                                match = re.search(r'-?\d+\.?\d*', val_str)
                                if match:
                                    card_data_attrs.append(f'data-{_normalize_col_name(col)}="{match.group()}"')
                        except:
                            pass
        
        data_attrs_str = ' ' + ' '.join(card_data_attrs) if card_data_attrs else ''
        html_parts.append(f'            <div class="data-card" data-row-index="{row_idx}"{data_attrs_str}>\n')
        html_parts.append('                <div class="card-header">\n')
        html_parts.append('                    <div class="card-header-main">\n')
        html_parts.append(f'                        <div class="card-ticker">{html.escape(ticker)}</div>\n')
        html_parts.append('                        <div class="card-price">\n')
        if current_price_val:
            price_str = str(current_price_val)
            # Remove $ if already present (from formatting)
            if price_str.startswith('$'):
                price_display = html.escape(price_str)
            else:
                price_display = '$' + html.escape(price_str)
            html_parts.append(f'                            <span class="card-price-value">{price_display}</span>\n')
        if change_pct_val:
            html_parts.append(f'                            <span class="card-price-change {change_class}">{html.escape(str(change_pct_val))}</span>\n')
        html_parts.append('                        </div>\n')
        html_parts.append('                    </div>\n')
        html_parts.append('                </div>\n')
        html_parts.append('                <div class="card-body">\n')
        
        # Primary metrics section (always visible)
        html_parts.append('                    <div class="card-primary">\n')
        
        # Add primary columns (skip ticker, current_price, change_pct as they're in header)
        primary_cols_to_show = ['strike_price', 'option_premium', 'expiration_date', 'daily_premium', 'net_daily_premium', 'net_daily_premi', 's_day_prem', 'short_daily_premium']
        shown_primary = set()
        
        for col_key in primary_cols_to_show:
            col_name = find_col_by_normalized(_normalize_col_name(col_key))
            if col_name and col_name in row.index:
                val = row[col_name]
                if pd.notna(val) and str(val).strip() and col_name not in shown_primary:
                    col_label = primary_columns_map.get(col_key, col_key.replace('_', ' ').title())
                    html_parts.append('                        <div class="card-primary-item">\n')
                    html_parts.append(f'                            <div class="card-primary-label">{html.escape(col_label)}</div>\n')
                    html_parts.append(f'                            <div class="card-primary-value">{html.escape(str(val))}</div>\n')
                    html_parts.append('                        </div>\n')
                    shown_primary.add(col_name)
        
        html_parts.append('                    </div>\n')
        
        # Expandable details section
        html_parts.append(f'                    <div class="card-details" id="{prefix}cardDetails_{row_idx}">\n')
        
        # Group columns by category for better organization
        option_cols = []
        greeks_cols = []
        premium_cols = []
        other_cols = []
        
        for col in df_display.columns:
            normalized = _normalize_col_name(col)
            # Skip columns already shown in primary section
            if normalized in ['ticker', 'current_price', 'curr_price', 'change_pct', 'price_with_change']:
                continue
            if col in shown_primary:
                continue
            
            val = row[col]
            
            # Skip always-hidden columns, but include regular hidden columns in cards
            is_always_hidden = normalized in always_hidden_columns_set
            if is_always_hidden:
                continue
            
            if pd.isna(val) or str(val).strip() == '':
                continue
            
            col_label = col.replace('_', ' ').title()
            
            # Categorize columns
            if any(x in normalized for x in ['strike', 'expiration', 'expiry', 'option_ticker', 'bid', 'ask']):
                option_cols.append((col, col_label, val))
            elif any(x in normalized for x in ['delta', 'theta', 'gamma', 'vega', 'iv', 'implied']):
                greeks_cols.append((col, col_label, val))
            elif any(x in normalized for x in ['premium', 'prem', 'net', 'total', 'daily']):
                premium_cols.append((col, col_label, val))
            else:
                other_cols.append((col, col_label, val))
        
        # Render categorized sections
        if option_cols:
            html_parts.append('                        <div class="card-section">\n')
            html_parts.append('                            <div class="card-section-title">Option Details</div>\n')
            for col, col_label, val in option_cols:
                html_parts.append('                            <div class="card-row">\n')
                html_parts.append(f'                                <span class="card-label">{html.escape(col_label)}</span>\n')
                html_parts.append(f'                                <span class="card-value">{html.escape(str(val))}</span>\n')
                html_parts.append('                            </div>\n')
            html_parts.append('                        </div>\n')
        
        if greeks_cols:
            html_parts.append('                        <div class="card-section">\n')
            html_parts.append('                            <div class="card-section-title">Greeks</div>\n')
            for col, col_label, val in greeks_cols:
                html_parts.append('                            <div class="card-row">\n')
                html_parts.append(f'                                <span class="card-label">{html.escape(col_label)}</span>\n')
                html_parts.append(f'                                <span class="card-value">{html.escape(str(val))}</span>\n')
                html_parts.append('                            </div>\n')
            html_parts.append('                        </div>\n')
        
        if premium_cols:
            html_parts.append('                        <div class="card-section">\n')
            html_parts.append('                            <div class="card-section-title">Premium & Returns</div>\n')
            for col, col_label, val in premium_cols:
                html_parts.append('                            <div class="card-row">\n')
                html_parts.append(f'                                <span class="card-label">{html.escape(col_label)}</span>\n')
                html_parts.append(f'                                <span class="card-value">{html.escape(str(val))}</span>\n')
                html_parts.append('                            </div>\n')
            html_parts.append('                        </div>\n')
        
        if other_cols:
            html_parts.append('                        <div class="card-section">\n')
            html_parts.append('                            <div class="card-section-title">Other</div>\n')
            for col, col_label, val in other_cols:
                html_parts.append('                            <div class="card-row">\n')
                html_parts.append(f'                                <span class="card-label">{html.escape(col_label)}</span>\n')
                html_parts.append(f'                                <span class="card-value">{html.escape(str(val))}</span>\n')
                html_parts.append('                            </div>\n')
            html_parts.append('                        </div>\n')
        
        html_parts.append('                    </div>\n')
        
        # Toggle button
        html_parts.append(f'                    <button class="card-toggle" onclick="toggleCardDetails(\'{prefix}\', {row_idx})" id="{prefix}cardToggle_{row_idx}">\n')
        html_parts.append('                        <span>Show More Details</span>\n')
        html_parts.append('                        <span class="card-toggle-icon">▼</span>\n')
        html_parts.append('                    </button>\n')
        
        html_parts.append('                </div>\n')
        html_parts.append('            </div>\n')
    
        html_parts.append("""        </div>
        
        <div class="stats">
            <div class="stat-item">
                <div class="stat-value" id="{prefix}totalCount">""" + str(len(df_display)) + """</div>
                <div class="stat-label">Total Results</div>
            </div>
            <div class="stat-item">
                <div class="stat-value" id="{prefix}visibleCount">""" + str(len(df_display)) + """</div>
                <div class="stat-label">Visible Rows</div>
            </div>
        </div>
        </div>
""")
    
    return ''.join(html_parts).format(prefix=prefix)






