"""
Desktop table HTML generation.
"""

import pandas as pd
import html as html_escape
from typing import Optional, Dict, Tuple, List
from .config import (
    COLUMN_GROUPS, GROUP_ORDER, HIDDEN_COLUMNS, ALWAYS_HIDDEN_COLUMNS,
    COLUMNS_AFTER_DAILY_PREMIUM
)
from .data_processor import normalize_col_name

HTML_RICH_COLUMNS = {'theta', 'l_theta', 'long_theta'}


def find_column_in_group(df: pd.DataFrame, group_name: str, side: str) -> Optional[str]:
    """Find column name that matches a group definition.
    
    Args:
        df: DataFrame to search
        group_name: Name of column group
        side: 'short' or 'long'
        
    Returns:
        Column name if found, None otherwise
    """
    if group_name not in COLUMN_GROUPS:
        return None
    
    variations = COLUMN_GROUPS[group_name].get(side, [])
    for col in df.columns:
        if col in variations:
            return col
    return None


def build_column_groups(df: pd.DataFrame) -> Tuple[Dict[str, Tuple[str, str]], Dict[str, str]]:
    """Build column groups mapping.
    
    Args:
        df: DataFrame with columns
        
    Returns:
        Tuple of (column_groups_dict, column_to_group_dict) where:
        - column_groups_dict: {group_name: (short_col, long_col)}
        - column_to_group_dict: {col_name: group_name}
    """
    column_groups = {}
    column_to_group = {}
    
    for group_name in GROUP_ORDER:
        if group_name not in COLUMN_GROUPS:
            continue
        
        short_col = find_column_in_group(df, group_name, 'short')
        long_col = find_column_in_group(df, group_name, 'long')
        
        if short_col and long_col:
            column_groups[group_name] = (short_col, long_col)
            column_to_group[short_col] = group_name
            column_to_group[long_col] = group_name
    
    return column_groups, column_to_group


def reorder_columns_for_display(df: pd.DataFrame, column_groups: Dict, column_to_group: Dict) -> List[str]:
    """Reorder columns for optimal display.
    
    Args:
        df: DataFrame
        column_groups: Column groups dictionary
        column_to_group: Column to group mapping
        
    Returns:
        List of column names in display order
    """
    original_cols = list(df.columns)
    new_order = []
    processed = set()
    
    # Add ungrouped columns first (excluding special ones)
    for col in original_cols:
        if col not in column_to_group:
            if col not in COLUMNS_AFTER_DAILY_PREMIUM:
                if col not in ['latest_option_writets', 'latest_opt_ts', 'price_change_pct']:
                    new_order.append(col)
                    processed.add(col)
    
    # Add grouped pairs
    for group_name in GROUP_ORDER:
        if group_name in column_groups:
            short_col, long_col = column_groups[group_name]
            if short_col not in processed and short_col in original_cols:
                new_order.append(short_col)
                processed.add(short_col)
            if long_col not in processed and long_col in original_cols:
                new_order.append(long_col)
                processed.add(long_col)
    
    # Add columns after daily premium
    for col in COLUMNS_AFTER_DAILY_PREMIUM:
        if col not in processed and col in original_cols:
            new_order.append(col)
            processed.add(col)
    
    # Add price_change_pct and latest_option_writets at the end (but before latest_opt_ts)
    for col in ['price_change_pct', 'latest_option_writets']:
        if col not in processed and col in original_cols:
            new_order.append(col)
            processed.add(col)
    
    # Add any remaining columns (except latest_opt_ts which goes last)
    for col in original_cols:
        if col not in processed and col != 'latest_opt_ts':
            new_order.append(col)
            processed.add(col)
    
    # Add latest_opt_ts as the rightmost column
    if 'latest_opt_ts' in original_cols and 'latest_opt_ts' not in processed:
        new_order.append('latest_opt_ts')
        processed.add('latest_opt_ts')
    
    return new_order


def build_table_html(
    df_display: pd.DataFrame,
    df_raw: pd.DataFrame,
    prefix: str
) -> str:
    """Build complete table HTML.
    
    Args:
        df_display: Formatted DataFrame for display
        df_raw: Raw DataFrame for filtering/sorting
        prefix: Tab prefix ('calls' or 'puts')
        
    Returns:
        Table HTML as string
    """
    if df_display.empty:
        return '<div style="padding: 20px; text-align: center;"><p>No data available.</p></div>'
    
    # Build column groups
    column_groups, column_to_group = build_column_groups(df_display)
    
    # Reorder columns
    new_order = reorder_columns_for_display(df_display, column_groups, column_to_group)
    df_display = df_display[new_order]
    if not df_raw.empty:
        df_raw = df_raw[[col for col in new_order if col in df_raw.columns]]
    
    # Build hidden columns sets
    hidden_set = {normalize_col_name(col) for col in HIDDEN_COLUMNS}
    always_hidden_set = {normalize_col_name(col) for col in ALWAYS_HIDDEN_COLUMNS}
    
    # Build HTML
    html_parts = []
    
    # Filter buttons
    html_parts.append("""        <div style="margin-bottom: 15px; display: flex; justify-content: space-between; gap: 10px; align-items: center;">
            <div>
                <button class="filter-button clear" onclick="toggleHiddenColumns('{prefix}')" id="{prefix}toggleHiddenBtn">
                    👁️ Show hidden columns
                </button>
            </div>
            <div style="text-align: right;">
                <button class="filter-button" onclick="toggleFilterSection('{prefix}')" id="{prefix}toggleFilterBtn">
                    🔍 Filter
                </button>
            </div>
        </div>
""".format(prefix=prefix))
    
    # Filter section (hidden by default - CSS handles display)
    html_parts.append(f"""        <div class="filter-section" id="{prefix}filterSection">
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
""")
    
    # Table wrapper
    html_parts.append(f'        <div class="table-wrapper hide-hidden" id="{prefix}tableWrapper">\n')
    html_parts.append(f'            <table id="{prefix}resultsTable">\n')
    html_parts.append("                <thead>\n")
    
    # Build header rows
    html_parts.extend(build_table_headers(df_display, column_groups, column_to_group, hidden_set, always_hidden_set, prefix))
    
    html_parts.append("                </thead>\n")
    html_parts.append("                <tbody>\n")
    
    # Build table rows
    html_parts.extend(build_table_rows(df_display, df_raw, hidden_set, always_hidden_set, prefix))
    
    html_parts.append("                </tbody>\n")
    html_parts.append("            </table>\n")
    html_parts.append("        </div>\n")
    
    return ''.join(html_parts)


def build_table_headers(
    df: pd.DataFrame,
    column_groups: Dict,
    column_to_group: Dict,
    hidden_set: set,
    always_hidden_set: set,
    prefix: str
) -> List[str]:
    """Build table header rows.
    
    Returns:
        List of HTML strings for header rows
    """
    html_parts = []
    processed_cols = set()
    
    # First row: Group headers
    html_parts.append('                    <tr class="group-header-row">\n')
    for col in df.columns:
        if col in processed_cols:
            continue
        
        if col in column_to_group:
            group_name = column_to_group[col]
            short_col, long_col = column_groups[group_name]
            if col == short_col:
                # Create group header
                display_name = COLUMN_GROUPS[group_name]['display_name']
                html_parts.append(f'                        <th class="group-header" colspan="2">{html_escape.escape(display_name)}</th>\n')
                processed_cols.add(short_col)
                processed_cols.add(long_col)
        else:
            # Regular column - empty cell in first row
            normalized = normalize_col_name(col)
            hidden_class = ' is-hidden-col' if normalized in hidden_set else ''
            always_hidden_class = ' always-hidden' if normalized in always_hidden_set else ''
            html_parts.append(f'                        <th class="group-header{hidden_class}{always_hidden_class}" colspan="1" style="background: transparent; border: none;"></th>\n')
            processed_cols.add(col)
    
    html_parts.append("                    </tr>\n")
    
    # Second row: Individual column headers
    html_parts.append('                    <tr class="column-header-row">\n')
    for col_idx, col in enumerate(df.columns):
        normalized = normalize_col_name(col)
        is_hidden = normalized in hidden_set
        is_always_hidden = normalized in always_hidden_set
        hidden_class = ' is-hidden-col' if is_hidden else ''
        always_hidden_class = ' always-hidden' if is_always_hidden else ''
        
        # Determine if grouped and which side
        grouped_class = ''
        short_long_label = ''
        if col in column_to_group:
            grouped_class = ' grouped-column'
            group_name = column_to_group[col]
            short_col, long_col = column_groups[group_name]
            if col == short_col:
                short_long_label = 'Short'
            elif col == long_col:
                short_long_label = 'Long'
        
        filterable_name = html_escape.escape(str(col))
        display_name = str(col)[:15]  # Truncate for display
        
        html_parts.append(f'                        <th class="sortable{grouped_class}{hidden_class}{always_hidden_class}" onclick="sortTable(\'{prefix}\', {col_idx})" data-filterable-name="{filterable_name}">')
        html_parts.append(f'                            <span class="column-name-display">{html_escape.escape(display_name)}</span>')
        if short_long_label:
            html_parts.append(f'                            <span class="column-name-short-long">{short_long_label}</span>')
        html_parts.append(f'                            <span class="column-name-filterable">{filterable_name}</span>')
        html_parts.append('                        </th>\n')
    
    html_parts.append("                    </tr>\n")
    
    return html_parts


def build_table_rows(
    df_display: pd.DataFrame,
    df_raw: pd.DataFrame,
    hidden_set: set,
    always_hidden_set: set,
    prefix: str
) -> List[str]:
    """Build table body rows.
    
    Returns:
        List of HTML strings for table rows
    """
    html_parts = []
    
    theta_pct_map = df_raw.attrs.get('theta_pct', {}) or {}
    l_theta_pct_map = df_raw.attrs.get('l_theta_pct', {}) or {}
    
    for row_idx, row in df_display.iterrows():
        html_parts.append(f'                    <tr data-row-index="{row_idx}">\n')
        
        for col in df_display.columns:
            cell_value = str(row[col]) if pd.notna(row[col]) else ''
            normalized_col = normalize_col_name(col)
            is_rich_html_cell = False
            rich_html_value = ''
            
            # Enhanced display for theta columns (include percent)
            if normalized_col in HTML_RICH_COLUMNS:
                pct_map = theta_pct_map if normalized_col == 'theta' else l_theta_pct_map
                pct_value = pct_map.get(row_idx)
                if pct_value is not None and pd.notna(pct_value):
                    pct_display = f"{pct_value:.2f}%"
                    main_display = cell_value.strip()
                    main_html = html_escape.escape(main_display) if main_display else '&nbsp;'
                    pct_html = html_escape.escape(pct_display)
                    rich_html_value = (
                        f'<div class="cell-main">{main_html}</div>'
                        f'<div class="cell-sub theta-percent">{pct_html}</div>'
                    )
                    is_rich_html_cell = True
                else:
                    cell_value = cell_value.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            else:
                cell_value = cell_value.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            
            # Get raw value for filtering
            raw_value = None
            if row_idx in df_raw.index and col in df_raw.columns:
                raw_val = df_raw.loc[row_idx, col]
                # Override raw value for theta columns to use percentage
                if normalized_col in HTML_RICH_COLUMNS:
                    pct_map = theta_pct_map if normalized_col == 'theta' else l_theta_pct_map
                    pct_value = pct_map.get(row_idx)
                    if pct_value is not None and pd.notna(pct_value):
                        raw_val = pct_value
                if pd.notna(raw_val):
                    try:
                        float_val = float(raw_val)
                        raw_value = str(float_val)
                    except (ValueError, TypeError):
                        raw_value = str(raw_val)
            
            # Build cell attributes
            hidden_class = ' is-hidden-col' if normalized_col in hidden_set else ''
            always_hidden_class = ' always-hidden' if normalized_col in always_hidden_set else ''
            all_classes = (hidden_class + always_hidden_class).strip()
            
            # Price change color class
            price_class = ''
            if normalized_col in ['change_pct', 'price_with_change'] and cell_value:
                if '+$' in cell_value or '(+' in cell_value:
                    price_class = ' price-positive'
                elif '-$' in cell_value or '(-' in cell_value:
                    price_class = ' price-negative'
            
            class_attr = f' class="{all_classes}{price_class}"' if (all_classes or price_class) else ''
            data_attr = f' data-raw="{html_escape.escape(str(raw_value))}"' if raw_value else ''
            
            # Make ticker column a link
            if normalized_col == 'ticker' and cell_value and cell_value.strip():
                ticker_value = cell_value.strip()
                ticker_link = f'/stock_info/{html_escape.escape(ticker_value)}'
                cell_content = f'<a href="{ticker_link}" target="_blank" style="color: #667eea; text-decoration: none; font-weight: 500;">{html_escape.escape(ticker_value)}</a>'
                html_parts.append(f'                        <td{class_attr}{data_attr}>{cell_content}</td>\n')
            elif is_rich_html_cell:
                html_parts.append(f'                        <td{class_attr}{data_attr}>{rich_html_value}</td>\n')
            else:
                html_parts.append(f'                        <td{class_attr}{data_attr}>{cell_value}</td>\n')
        
        html_parts.append("                    </tr>\n")
    
    return html_parts

