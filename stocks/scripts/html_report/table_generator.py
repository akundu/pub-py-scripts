"""
Table HTML generation for HTML report.
"""

import pandas as pd
import html
import re
from datetime import datetime
from .formatters import truncate_header, normalize_col_name
from .constants import (
    HIDDEN_COLUMNS, ALWAYS_HIDDEN_COLUMNS, COLUMN_GROUPS, 
    GROUP_DISPLAY_NAMES, GROUP_ORDER, COLUMNS_AFTER_DAILY_PREMIUM,
    PRIMARY_CARD_COLUMNS, PRIMARY_COLUMN_LABELS, DESIRED_COLUMN_ORDER,
    COLUMN_VARIATIONS
)

def generate_table_and_cards_html(df_display: pd.DataFrame, df_raw: pd.DataFrame, prefix: str, is_active: bool, normalize_col_name_func=None) -> str:
    """Generate HTML for table, filters, and cards for a given DataFrame and prefix.
    
    Args:
        df_display: Formatted DataFrame for display
        df_raw: Raw DataFrame with original values for filtering
        prefix: Prefix for IDs ('calls' or 'puts')
        is_active: Whether this tab should be active by default
        normalize_col_name_func: Optional function to normalize column names (uses imported if None)
        
    Returns:
        String containing HTML for the tab content
    """
    # Use provided function or imported one
    if normalize_col_name_func is None:
        normalize_col_name_func = normalize_col_name
    
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

    hidden_columns_set = set(normalize_col_name_func(col) for col in hidden_columns_list)
    
    # Columns to always hide (hidden in all cases, even when "Show hidden columns" is clicked)
    always_hidden_columns_list = [
        'l_cnt_avl',
        'long_contracts_available',  # Also hide the full name variation
    ]
    
    always_hidden_columns_set = set(normalize_col_name_func(col) for col in always_hidden_columns_list)

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
            normalized_col = normalize_col_name_func(col)
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
            normalized_col = normalize_col_name_func(col)
            
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
            if normalize_col_name_func(col) == target_normalized:
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
                    if normalize_col_name_func(c) == normalize_col_name_func(col):
                        col_found = c
                        break
                if col_found and col_found in df_raw.columns:
                    raw_val = df_raw.loc[row_idx, col_found]
                    if pd.notna(raw_val):
                        try:
                            # Try to get numeric value
                            if isinstance(raw_val, (int, float)):
                                card_data_attrs.append(f'data-{normalize_col_name_func(col)}="{raw_val}"')
                            else:
                                val_str = str(raw_val)
                                # Try to extract number
                                match = re.search(r'-?\d+\.?\d*', val_str)
                                if match:
                                    card_data_attrs.append(f'data-{normalize_col_name_func(col)}="{match.group()}"')
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
        </div>
""")
    
    return ''.join(html_parts).format(prefix=prefix)





