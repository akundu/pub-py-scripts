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
        
        /* Hidden columns handling */
        .table-wrapper.hide-hidden th.is-hidden-col,
        .table-wrapper.hide-hidden td.is-hidden-col {
            display: none;
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
        
        .column-name-filterable {
            display: none;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
            color: #667eea;
        }
        
        th.showing-filterable .column-name-display {
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
            padding: 12px;
            border-bottom: 1px solid #dee2e6;
            color: #495057;
            max-width: 15ch;
            white-space: normal;
            word-wrap: break-word;
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
            
            .filter-controls {
                flex-direction: column;
            }
            
            .filter-input-group {
                width: 100%;
            }
        }
"""


def get_javascript():
    """Get JavaScript code for table sorting and filtering functionality.
    
    Returns:
        String containing JavaScript code
    """
    return """        let sortDirection = {};
        let currentSortColumn = -1;
        let activeFilters = [];
        let filterLogic = 'AND';
        
        // Column name mapping (original names to display names)
        const columnMap = {};
        
        // Initialize column map
        function initColumnMap() {
            const table = document.getElementById('resultsTable');
            const headers = table.querySelectorAll('th');
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
        function evaluateMathExpression(expression, row) {
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
                
                const colIndex = findColumnIndex(field);
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
        function getAllColumnNames() {
            const table = document.getElementById('resultsTable');
            const headers = Array.from(table.querySelectorAll('th'));
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
                        columnNames.push(headerText);
                    }
                }
            });
            return columnNames;
        }
        
        // Find column index by field name (supports substring matching)
        function findColumnIndex(fieldName) {
            const table = document.getElementById('resultsTable');
            const headers = Array.from(table.querySelectorAll('th'));
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
        function parseFilterExpression(expression) {
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
                            const valueColIndex = findColumnIndex(valueStr);
                            if (valueColIndex >= 0) {
                                // Math expression compared to another field: "curr_price*1.05 < strike_price"
                                return {
                                    field: fieldExpr,
                                    operator: op,
                                    value: valueStr,
                                    isFieldComparison: true,
                                    hasMath: true
                                };
                            } else {
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
                        }
                        
                        // Check if value is a field name (field-to-field comparison without math)
                        const valueColIndex = findColumnIndex(valueStr);
                        if (valueColIndex >= 0) {
                            return {
                                field: fieldExpr,
                                operator: op,
                                value: valueStr,
                                isFieldComparison: true
                            };
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
        function evaluateFilter(filter, row) {
            // Handle exists/not_exists (only works for single fields, not expressions)
            if (filter.operator === 'exists' || filter.operator === 'not_exists') {
                if (filter.hasMath) return false; // Can't use exists/not_exists with math expressions
                const colIndex = findColumnIndex(filter.field);
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
                const leftValue = evaluateMathExpression(filter.field, row);
                if (leftValue === null) {
                    console.warn('Could not evaluate math expression:', filter.field);
                    return false; // Can't evaluate expression
                }
                
                // Get the right side value
                let rightValue = null;
                if (filter.isFieldComparison) {
                    // Right side is also a field: "curr_price*1.05 < strike_price"
                    const valueColIndex = findColumnIndex(filter.value);
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
                const colIndex = findColumnIndex(filter.field);
                if (colIndex < 0) {
                    console.warn('Column not found for filter field:', filter.field);
                    return false;
                }
                const cell = row.cells[colIndex];
                if (!cell) return false;
                
                const valueColIndex = findColumnIndex(filter.value);
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
            const colIndex = findColumnIndex(filter.field);
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
        function applyFilters() {
            const table = document.getElementById('resultsTable');
            const tbody = table.querySelector('tbody');
            const rows = Array.from(tbody.querySelectorAll('tr'));
            const errorDiv = document.getElementById('filterError');
            
            if (errorDiv) errorDiv.textContent = '';
            
            if (activeFilters.length === 0) {
                rows.forEach(row => {
                    row.style.display = '';
                });
                updateVisibleCount();
                applyRowStriping();
                return;
            }
            
            rows.forEach(row => {
                let matches = activeFilters.map(filter => evaluateFilter(filter, row));
                
                let shouldShow;
                if (filterLogic === 'AND') {
                    shouldShow = matches.every(m => m);
                } else {
                    shouldShow = matches.some(m => m);
                }
                
                row.style.display = shouldShow ? '' : 'none';
            });
            
            updateVisibleCount();
            applyRowStriping();
        }
        
        // Validate filter fields exist
        function validateFilter(filter) {
            const errorMessages = [];
            
            // Check main field
            if (filter.field && !filter.hasMath) {
                // For non-math expressions, check if field exists
                const colIndex = findColumnIndex(filter.field);
                if (colIndex < 0) {
                    const availableColumns = getAllColumnNames();
                    const suggestions = availableColumns.filter(col => 
                        col.toLowerCase().includes(filter.field.toLowerCase()) || 
                        filter.field.toLowerCase().includes(col.toLowerCase())
                    ).slice(0, 5);
                    
                    let errorMsg = `Column "${filter.field}" not found. `;
                    if (suggestions.length > 0) {
                        errorMsg += `Did you mean: ${suggestions.join(', ')}? `;
                    }
                    errorMsg += `Available columns: ${availableColumns.slice(0, 10).join(', ')}${availableColumns.length > 10 ? '...' : ''}`;
                    errorMessages.push(errorMsg);
                }
            } else if (filter.hasMath) {
                // For math expressions, extract field names and validate them
                const fieldPattern = /[a-zA-Z_][a-zA-Z0-9_]*/g;
                const fields = filter.field.match(fieldPattern) || [];
                const jsKeywords = ['if', 'else', 'for', 'while', 'function', 'return', 'var', 'let', 'const', 'true', 'false', 'null', 'undefined'];
                const validFields = fields.filter(f => !jsKeywords.includes(f.toLowerCase()));
                
                for (const field of validFields) {
                    const colIndex = findColumnIndex(field);
                    if (colIndex < 0) {
                        errorMessages.push(`Field "${field}" in expression "${filter.field}" not found.`);
                    }
                }
            }
            
            // Check comparison field if it's a field-to-field comparison
            if (filter.isFieldComparison && filter.value) {
                const valueColIndex = findColumnIndex(filter.value);
                if (valueColIndex < 0) {
                    const availableColumns = getAllColumnNames();
                    const suggestions = availableColumns.filter(col => 
                        col.toLowerCase().includes(String(filter.value).toLowerCase()) || 
                        String(filter.value).toLowerCase().includes(col.toLowerCase())
                    ).slice(0, 5);
                    
                    let errorMsg = `Column "${filter.value}" not found for comparison. `;
                    if (suggestions.length > 0) {
                        errorMsg += `Did you mean: ${suggestions.join(', ')}?`;
                    }
                    errorMessages.push(errorMsg);
                }
            }
            
            return errorMessages;
        }
        
        // Add filter
        function addFilter() {
            const input = document.getElementById('filterInput');
            const expression = input.value.trim();
            const errorDiv = document.getElementById('filterError');
            
            if (errorDiv) errorDiv.textContent = '';
            
            if (!expression) return;
            
            const filter = parseFilterExpression(expression);
            if (!filter) {
                if (errorDiv) {
                    errorDiv.textContent = 'Invalid filter expression. Format: field operator value (e.g., "pe_ratio > 20", "volume exists")';
                }
                return;
            }
            
            // Validate filter fields exist
            const validationErrors = validateFilter(filter);
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
            
            activeFilters.push(filter);
            input.value = '';
            
            // Update filter display
            updateFilterDisplay();
            applyFilters();
            updateURL();
        }
        
        // Remove filter
        function removeFilter(index) {
            activeFilters.splice(index, 1);
            updateFilterDisplay();
            applyFilters();
            updateURL();
        }
        
        // Clear all filters
        function clearFilters() {
            activeFilters = [];
            updateFilterDisplay();
            applyFilters();
            updateURL();
        }
        
        // Update filter display
        function updateFilterDisplay() {
            const container = document.getElementById('activeFilters');
            if (!container) return;
            
            container.innerHTML = '';
            
            if (activeFilters.length === 0) {
                container.innerHTML = '<p style="color: #6c757d; font-style: italic;">No active filters</p>';
                return;
            }
            
            activeFilters.forEach((filter, index) => {
                const filterDiv = document.createElement('div');
                filterDiv.style.cssText = 'display: inline-block; margin: 5px; padding: 5px 10px; background: #667eea; color: white; border-radius: 5px; font-size: 0.9em;';
                
                const filterText = document.createElement('span');
                filterText.textContent = `${filter.field} ${filter.operator} ${filter.value !== null ? filter.value : ''}`;
                filterDiv.appendChild(filterText);
                
                const removeBtn = document.createElement('button');
                removeBtn.textContent = '×';
                removeBtn.style.cssText = 'margin-left: 8px; background: rgba(255,255,255,0.3); border: none; color: white; cursor: pointer; border-radius: 3px; padding: 2px 6px;';
                removeBtn.onclick = () => removeFilter(index);
                filterDiv.appendChild(removeBtn);
                
                container.appendChild(filterDiv);
            });
        }
        
        // Update filter logic
        function updateFilterLogic(logic) {
            filterLogic = logic;
            applyFilters();
            updateURL();
        }
        
        // Update visible count
        function updateVisibleCount() {
            const table = document.getElementById('resultsTable');
            const tbody = table.querySelector('tbody');
            const visibleRows = Array.from(tbody.querySelectorAll('tr')).filter(row => row.style.display !== 'none');
            const visibleCountEl = document.getElementById('visibleCount');
            if (visibleCountEl) {
                visibleCountEl.textContent = visibleRows.length;
            }
        }
        
        // Apply row striping based on visible rows
        function applyRowStriping() {
            const table = document.getElementById('resultsTable');
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
        function applyColumnStriping() {
            const table = document.getElementById('resultsTable');
            const thead = table.querySelector('thead');
            const tbody = table.querySelector('tbody');
            const wrapper = document.getElementById('tableWrapper');
            
            if (!thead || !tbody) return;
            
            const headers = Array.from(thead.querySelectorAll('th'));
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
                // 1. It doesn't have is-hidden-col class, OR
                // 2. It has is-hidden-col but hide-hidden is false (showing all columns)
                const isHiddenCol = th.classList.contains('is-hidden-col');
                if (!isHiddenCol || !hideHidden) {
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
        
        function sortTable(columnIndex) {
            const table = document.getElementById('resultsTable');
            const tbody = table.querySelector('tbody');
            const rows = Array.from(tbody.querySelectorAll('tr:not([style*="display: none"])'));
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
            
            // Check if sorting by change_pct (renamed from price_with_change)
            // We can sort directly using the data-raw attribute which contains the percentage value
            // No need to find a separate column - the change_pct column itself has the percentage in data-raw
            const currentHeader = headers[columnIndex];
            const filterableName = currentHeader.getAttribute('data-filterable-name');
            let sortColumnIndex = columnIndex;
            // For change_pct column, we'll use its own data-raw attribute which contains the percentage
            // The JavaScript sorting code below will automatically use data-raw if available
            
            // Sort rows
            rows.sort((a, b) => {
                const aCell = a.cells[sortColumnIndex];
                const bCell = b.cells[sortColumnIndex];
                if (!aCell || !bCell) return 0;
                
                const aRaw = aCell.getAttribute('data-raw');
                const bRaw = bCell.getAttribute('data-raw');
                
                let aNum = aRaw !== null ? parseFloat(aRaw) : NaN;
                let bNum = bRaw !== null ? parseFloat(bRaw) : NaN;
                
                if (isNaN(aNum)) {
                    aNum = parseFloat(aCell.textContent.trim().replace(/[^0-9.-]/g, ''));
                }
                if (isNaN(bNum)) {
                    bNum = parseFloat(bCell.textContent.trim().replace(/[^0-9.-]/g, ''));
                }
                
                let comparison = 0;
                
                if (!isNaN(aNum) && !isNaN(bNum)) {
                    comparison = aNum - bNum;
                } else {
                    const aText = aCell.textContent.trim();
                    const bText = bCell.textContent.trim();
                    comparison = aText.localeCompare(bText);
                }
                
                return sortDirection[columnIndex] === 'asc' ? comparison : -comparison;
            });
            
            // Re-append sorted rows (only visible ones)
            const hiddenRows = Array.from(tbody.querySelectorAll('tr[style*="display: none"]'));
            rows.forEach(row => tbody.appendChild(row));
            hiddenRows.forEach(row => tbody.appendChild(row));
            
            updateVisibleCount();
            applyRowStriping();
        }
        
        // Tab switching functionality
        function switchTab(tabIndex) {
            const tabContents = document.querySelectorAll('.tab-content');
            tabContents.forEach(content => content.classList.remove('active'));
            
            const tabButtons = document.querySelectorAll('.tab-button');
            tabButtons.forEach(button => button.classList.remove('active'));
            
            tabContents[tabIndex].classList.add('active');
            tabButtons[tabIndex].classList.add('active');
        }
        
        // Handle Enter key in filter input
        function handleFilterKeyPress(event) {
            if (event.key === 'Enter') {
                addFilter();
            }
        }
        
        // Toggle filter section and column names
        function toggleFilterSection() {
            const filterSection = document.getElementById('filterSection');
            const button = document.getElementById('toggleFilterBtn');
            const headers = document.querySelectorAll('#resultsTable th');
            
            if (!filterSection || !button) return;
            
            // Toggle filter section
            const isExpanded = filterSection.classList.contains('expanded');
            
            if (isExpanded) {
                // Collapse: hide filter section and show display names
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
        function toggleHiddenColumns() {
            const wrapper = document.getElementById('tableWrapper');
            const btn = document.getElementById('toggleHiddenBtn');
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
            applyRowStriping();
            applyColumnStriping();
        }
        
        // Update URL with current filters
        function updateURL() {
            const params = new URLSearchParams();
            
            // Add filter logic (only if not default AND)
            if (filterLogic && filterLogic !== 'AND') {
                params.set('filterLogic', filterLogic);
            }
            
            // Add filters
            if (activeFilters.length > 0) {
                const filterStrings = activeFilters.map(f => {
                    let filterStr = f.field;
                    if (f.operator) {
                        filterStr += ' ' + f.operator;
                        if (f.value !== null) {
                            filterStr += ' ' + f.value;
                        }
                    }
                    return filterStr;
                });
                params.set('filters', filterStrings.join('|'));
            }
            
            // Update URL without reloading page
            const newURL = window.location.pathname + (params.toString() ? '?' + params.toString() : '');
            window.history.replaceState({}, '', newURL);
        }
        
        // Load filters from URL
        function loadFiltersFromURL() {
            const params = new URLSearchParams(window.location.search);
            const errorDiv = document.getElementById('filterError');
            if (errorDiv) {
                errorDiv.textContent = '';
                errorDiv.style.display = 'none';
            }
            
            // Load filter logic
            const urlFilterLogic = params.get('filterLogic');
            if (urlFilterLogic && (urlFilterLogic === 'AND' || urlFilterLogic === 'OR')) {
                filterLogic = urlFilterLogic;
                // Update radio button
                const radio = document.querySelector(`input[name="filterLogic"][value="${filterLogic}"]`);
                if (radio) {
                    radio.checked = true;
                }
            }
            
            // Load filters
            const filtersParam = params.get('filters');
            if (filtersParam) {
                const filterStrings = filtersParam.split('|');
                activeFilters = [];
                const validationErrors = [];
                
                for (const filterStr of filterStrings) {
                    if (filterStr.trim()) {
                        const filter = parseFilterExpression(filterStr.trim());
                        if (filter) {
                            // Validate filter before adding
                            const errors = validateFilter(filter);
                            if (errors.length > 0) {
                                validationErrors.push(...errors);
                            } else {
                                activeFilters.push(filter);
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
                if (activeFilters.length > 0) {
                    const filterSection = document.getElementById('filterSection');
                    const button = document.getElementById('toggleFilterBtn');
                    const headers = document.querySelectorAll('#resultsTable th');
                    
                    if (filterSection && button) {
                        filterSection.classList.add('expanded');
                        headers.forEach(th => {
                            th.classList.add('showing-filterable');
                        });
                        button.textContent = '✖️ Hide Filter';
                        button.title = 'Hide filter options and show display column names';
                    }
                }
                
                updateFilterDisplay();
                applyFilters();
            }
        }
        
        // Calculate and display time ago
        function updateTimeAgo() {
            const timeAgoEl = document.getElementById('timeAgo');
            const generatedTimeEl = document.getElementById('generatedTime');
            if (!timeAgoEl || !generatedTimeEl) return;
            
            // Get the ISO timestamp from the data-generated attribute on the paragraph
            const generatedTimeStr = generatedTimeEl.getAttribute('data-generated');
            if (!generatedTimeStr) return;
            
            try {
                const generatedTime = new Date(generatedTimeStr);
                // Check if date is valid
                if (isNaN(generatedTime.getTime())) {
                    console.error('Invalid date string:', generatedTimeStr);
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
        
        // Initialize
        document.addEventListener('DOMContentLoaded', function() {
            switchTab(0);
            initColumnMap();
            updateFilterDisplay();
            loadFiltersFromURL();
            updateTimeAgo(); // Update time ago on page load
            // Update time ago every minute
            setInterval(updateTimeAgo, 60000);
            // Ensure hidden columns are hidden on load
            const wrapper = document.getElementById('tableWrapper');
            const toggleBtn = document.getElementById('toggleHiddenBtn');
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
            applyRowStriping();
            applyColumnStriping();
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
    
    # Store raw values before formatting (for filtering) - keep original DataFrame
    df_raw = df.copy()
    
    # Format numeric columns for better display
    numeric_cols = [
        'ticker','pe_ratio','market_cap_b','curr_price','strike_price','price_above_curr','opt_prem.','IV','delta','theta','expiration_date','days_to_expiry','s_prem_tot','s_day_prem','l_strike','l_prem','liv','l_delta','l_theta','l_expiration_date','l_days_to_expiry','l_prem_tot','l_cnt_avl','prem_diff','net_premium','net_daily_premi','volume','num_contracts','option_ticker','l_option_ticker',
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
                if isinstance(x, pd.Timestamp):
                    return x.strftime('%Y-%m-%d')
                elif isinstance(x, (str, datetime)):
                    dt = pd.to_datetime(x, errors='coerce')
                    if pd.notna(dt):
                        return dt.strftime('%Y-%m-%d')
                    else:
                        # Try to extract date portion from string
                        x_str = str(x)
                        if ' ' in x_str:
                            return x_str.split(' ')[0]
                        elif 'T' in x_str:
                            return x_str.split('T')[0]
                        elif len(x_str) >= 10:
                            return x_str[:10]
            except (ValueError, TypeError, AttributeError):
                # If all parsing fails, try to extract first 10 characters
                x_str = str(x)
                if len(x_str) >= 10:
                    return x_str[:10]
        return x
    
    # Format date columns first
    for col in df_display.columns:
        normalized_col = _normalize_col_name(col)
        if normalized_col in ['expiration_date', 'l_expiration_date', 'long_expiration_date']:
            df_display[col] = df_display[col].apply(lambda x: format_date_value(x, col))
    
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
        'ticker', 'pe_ratio', 'market_cap_b', 'current_price', 'price_with_change',
        'strike_price', 'option_premium', 'expiration_date', 'bid_ask', 'delta', 'theta',
        'short_premium_total', 'short_daily_premium',
        'long_strike_price', 'long_option_premium', 'long_expiration_date', 'long_bid_ask', 'long_delta', 'long_theta',
        'long_implied_volatility', 'long_days_to_expiry', 
        'long_premium_total', 'long_contracts_available',
        'net_premium', 'net_daily_premium',
        'price_above_current', 'premium_above_diff_percentage',
        'implied_volatility', 'days_to_expiry',
        'potential_premium', 'daily_premium',
        'volume', 'num_contracts', 'write_timestamp', 'option_ticker', 'long_option_ticker',
        'premium_diff',  # Will be hidden by default
        'spread_slippage', 'net_premium_after_spread', 'net_daily_premium_after_spread',
        'spread_impact_pct', 'liquidity_score', 'assignment_risk', 'trade_quality'
    ]
    
    # Get existing columns in desired order, handling variations
    ordered_cols = []
    for desired_name in desired_order_names:
        actual_col = find_matching_column(desired_name)
        if actual_col:
            ordered_cols.append(actual_col)
    
    # Add any remaining columns (excluding price_change_pct from visible)
    remaining_cols = [col for col in df_display.columns if col not in ordered_cols and col != 'price_change_pct']
    # Add price_change_pct at the end (hidden, used for sorting)
    if 'price_change_pct' in df_display.columns:
        remaining_cols.append('price_change_pct')
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
    
    # Apply compact headers to keep column names concise
    # compact_headers = {}
    # for col in df_display.columns:
    #     if col in COMPACT_HEADER_MAP:
    #         compact_headers[col] = COMPACT_HEADER_MAP[col]
    #     else:
    #         compact_headers[col] = col
    # df_display = df_display.rename(columns=compact_headers)
    
    # Get current timestamp for display
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
    # Get ISO timestamp with timezone for JavaScript parsing (ensure it's parseable)
    iso_timestamp = now.isoformat()
    
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
            <p id="generatedTime" data-generated="""" + iso_timestamp + """">Generated: """ + timestamp + """ <span id="timeAgo"></span></p>
            <p>Click column headers to sort • """ + str(len(df)) + """ total results</p>
        </div>
        
        <div class="tab-content active">
        <div style="margin-bottom: 15px; display: flex; justify-content: space-between; gap: 10px; align-items: center;">
            <div>
                <button class="filter-button clear" onclick="toggleHiddenColumns()" id="toggleHiddenBtn" title="Show or hide default-hidden columns">
                    👁️ Show hidden columns
                </button>
            </div>
            <div style="text-align: right;">
                <button class="filter-button" onclick="toggleFilterSection()" id="toggleFilterBtn" title="Show/hide filter options and filterable column names">
                    🔍 Filter
                </button>
            </div>
        </div>
        <div class="filter-section" id="filterSection">
            <h3 style="margin-top: 0; color: #667eea;">🔍 Filter Options</h3>
            <div class="filter-logic">
                <label>Filter Logic:</label>
                <label><input type="radio" name="filterLogic" value="AND" checked onchange="updateFilterLogic('AND')"> AND</label>
                <label><input type="radio" name="filterLogic" value="OR" onchange="updateFilterLogic('OR')"> OR</label>
            </div>
            <div class="filter-controls">
                <div class="filter-input-group">
                    <input type="text" id="filterInput" class="filter-input" placeholder="e.g., pe_ratio > 20, volume exists, net_daily_premium > 100" onkeypress="handleFilterKeyPress(event)">
                    <button class="filter-button" onclick="addFilter()">Add Filter</button>
                    <button class="filter-button clear" onclick="clearFilters()">Clear All</button>
                </div>
            </div>
            <div id="filterError" class="filter-error"></div>
            <div id="activeFilters" style="margin-top: 10px;"></div>
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
        <div class="table-wrapper hide-hidden" id="tableWrapper">
            <table id="resultsTable">
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
        'theta',
        'l_theta',
        'long_theta',
        'l_days_to_expiry',
        'long_days_to_expiry',
        'l_cnt_avl',
        'long_contracts_available',
        'option_ticker',
        'l_option_ticker',
        'net_premium_after_spread',
        'spread_slippage',
        'net_daily_premium_after_spread',
        'spread_impact_pct',
        'liquidity_score',
        'assignment_risk',
    ]

    hidden_columns_set = set(_normalize_col_name(col) for col in hidden_columns_list)

    # Generate table headers with filterable name toggle
    for col in df_display.columns:
        col_index = df_display.columns.get_loc(col)
        truncated_title = truncate_header(str(col), 15)
        # Use the original column name as the filterable name
        filterable_name = html.escape(str(col))
        normalized_col = _normalize_col_name(col)
        is_hidden = normalized_col in hidden_columns_set
        hidden_class = ' is-hidden-col' if is_hidden else ''
        html_parts.append(f'                        <th class="sortable{hidden_class}" onclick="sortTable({col_index})" data-filterable-name="{filterable_name}">')
        html_parts.append(f'                            <span class="column-name-display">{truncated_title}</span>')
        html_parts.append(f'                            <span class="column-name-filterable">{filterable_name}</span>')
        html_parts.append(f'                        </th>\n')
    
    html_parts.append("""                    </tr>
                </thead>
                <tbody>
""")
    
    # Generate table rows with raw values stored in data attributes
    for row_idx, row in df_display.iterrows():
        html_parts.append("                    <tr>\n")
        for col in df_display.columns:
            cell_value = str(row[col]) if pd.notna(row[col]) else ''
            # Escape HTML special characters
            cell_value = cell_value.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            
            # Get raw value for filtering
            raw_value = None
            raw_text = None
            if row_idx in df_raw.index and col in df_raw.columns:
                raw_val = df_raw.loc[row_idx, col]
                if pd.notna(raw_val):
                    # Store numeric value if it's a number
                    try:
                        # Try to convert to float first to check if it's numeric
                        float_val = float(raw_val)
                        # If successful and it's actually a number (not NaN), store as numeric
                        if not pd.isna(float_val):
                            raw_value = str(float_val)
                        else:
                            raw_text = str(raw_val)
                    except (ValueError, TypeError):
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
            
            # Build td with data attributes
            td_attrs = []
            if raw_value is not None:
                td_attrs.append(f'data-raw="{html.escape(str(raw_value))}"')
            if raw_text is not None:
                td_attrs.append(f'data-raw-text="{html.escape(str(raw_text))}"')
            # Hidden class for default-hidden columns
            normalized_col = _normalize_col_name(col)
            td_hidden_class = ' is-hidden-col' if normalized_col in hidden_columns_set else ''
            
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



