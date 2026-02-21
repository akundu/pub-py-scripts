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


def build_column_groups(df: pd.DataFrame) -> Tuple[Dict[str, Dict[str, List[str]]], Dict[str, str]]:
    """Build column groups mapping.
    
    Args:
        df: DataFrame with columns
        
    Returns:
        Tuple of (column_groups_dict, column_to_group_dict) where:
        - column_groups_dict: {group_name: {'short': [...], 'long': [...], 'extras': [...]} }
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
            group_info = {
                'short': [short_col],
                'long': [long_col],
                'extras': []
            }
            
            # Attach any extra columns defined for this group (e.g., premium_ratio_pct under option_premium)
            extras = COLUMN_GROUPS[group_name].get('extras', [])
            for col in df.columns:
                if col in extras:
                    group_info['extras'].append(col)
                    column_to_group[col] = group_name
            
            column_groups[group_name] = group_info
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
                if col not in ['latest_option_writets', 'price_change_pct']:
                    new_order.append(col)
                    processed.add(col)
    
    # Add grouped columns (short, long, then extras like premium_ratio_pct)
    for group_name in GROUP_ORDER:
        if group_name in column_groups:
            group_info = column_groups[group_name]
            # Short
            for col in group_info.get('short', []):
                if col not in processed and col in original_cols:
                    new_order.append(col)
                    processed.add(col)
            # Long
            for col in group_info.get('long', []):
                if col not in processed and col in original_cols:
                    new_order.append(col)
                    processed.add(col)
            # Extras (e.g., premium_ratio_pct)
            for col in group_info.get('extras', []):
                if col not in processed and col in original_cols:
                    new_order.append(col)
                    processed.add(col)
    
    # Add columns after daily premium
    for col in COLUMNS_AFTER_DAILY_PREMIUM:
        if col not in processed and col in original_cols:
            new_order.append(col)
            processed.add(col)
    
    # Add price_change_pct and latest_option_writets at the end
    # Note: latest_opt_ts is renamed to latest_option_writets in data_processor, so we only use latest_option_writets
    for col in ['price_change_pct', 'latest_option_writets']:
        if col not in processed and col in original_cols:
            new_order.append(col)
            processed.add(col)
    
    # Add any remaining columns (excluding latest_opt_ts since it's renamed to latest_option_writets)
    for col in original_cols:
        if col not in processed and col != 'latest_opt_ts':
            new_order.append(col)
            processed.add(col)
    
    return new_order


def build_table_html(
    df_display: pd.DataFrame,
    df_raw: pd.DataFrame,
    prefix: str,
    empty: bool = False
) -> str:
    """Build complete table HTML.
    
    Args:
        df_display: Formatted DataFrame for display (used for structure if empty=False)
        df_raw: Raw DataFrame for filtering/sorting (used for structure if empty=False)
        prefix: Tab prefix ('calls' or 'puts')
        empty: If True, create empty table structure (no rows, data loaded via API)
        
    Returns:
        Table HTML as string
    """
    if empty:
        # Create empty table structure - will be populated by JavaScript via API
        # We still need column structure, so use df_display's columns if available,
        # even when there are zero rows (static shell case).
        if len(df_display.columns) > 0:
            # Build proper table structure with headers from df_display
            column_groups, column_to_group = build_column_groups(df_display)
            new_order = reorder_columns_for_display(df_display, column_groups, column_to_group)
            df_display = df_display[new_order]
            hidden_set = {normalize_col_name(col) for col in HIDDEN_COLUMNS}
            always_hidden_set = {normalize_col_name(col) for col in ALWAYS_HIDDEN_COLUMNS}
            
            html_parts = []
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
                <strong>📋 All Filterable Fields:</strong><br>
                <strong>Stock Info:</strong> <code>ticker</code>, <code>current_price</code> (or <code>curr_price</code>), <code>pe_ratio</code>, <code>market_cap_b</code>, <code>price_change_pct</code>, <code>price_with_change</code><br>
                <strong>Strike Prices:</strong> <code>strike_price</code>, <code>l_strike</code> (long strike)<br>
                <strong>Expiration Dates:</strong> <code>expiration_date</code>, <code>l_expiration_date</code> (long expiration), <code>days_to_expiry</code>, <code>l_days_to_expiry</code><br>
                <strong>Option Premiums:</strong> <code>opt_prem.</code> (or <code>option_premium</code>), <code>l_prem</code> (or <code>l_opt_prem</code>), <code>premium_ratio_pct</code>, <code>s_prem_tot</code>, <code>l_prem_tot</code>, <code>net_premium</code><br>
                <strong>Bid/Ask:</strong> <code>bid:ask</code>, <code>l_bid:ask</code> (long bid/ask)<br>
                <strong>Greeks:</strong> <code>delta</code>, <code>l_delta</code> (long delta), <code>theta</code>, <code>l_theta</code> (long theta)<br>
                <strong>Daily Premiums:</strong> <code>s_day_prem</code>, <code>net_daily_premi</code> (or <code>net_daily_premium</code>), <code>net_daily_premium_after_spread</code><br>
                <strong>Trading:</strong> <code>volume</code>, <code>num_contracts</code>, <code>option_ticker</code>, <code>l_option_ticker</code><br>
                <strong>Spread Analysis:</strong> <code>spread</code> (virtual field - short leg spread), <code>l_spread</code> (virtual field - long leg spread), <code>spread_slippage</code>, <code>spread_impact_pct</code><br>
                <strong>Bid/Ask Virtual Fields:</strong> <code>bid</code> (virtual - extracts bid from bid:ask), <code>ask</code> (virtual - extracts ask from bid:ask), <code>l_bid</code> (virtual - extracts bid from l_bid:ask), <code>l_ask</code> (virtual - extracts ask from l_bid:ask)<br>
                <strong>Quality Metrics:</strong> <code>trade_quality</code>, <code>liquidity_score</code>, <code>assignment_risk</code><br>
                <strong>Other:</strong> <code>option_type</code>, <code>buy_cost</code>, <code>price_above_curr</code>, <code>premium_diff</code>, <code>latest_option_writets</code><br>
                <br>
                <strong>Filter Examples:</strong><br>
                • <code>pe_ratio > 20</code> - P/E ratio greater than 20<br>
                • <code>market_cap_b < 3.5</code> - Market cap less than 3.5B<br>
                • <code>volume exists</code> - Volume data exists<br>
                • <code>net_daily_premium > 100</code> - Net daily premium greater than 100<br>
                • <code>delta < 0.5</code> - Delta less than 0.5<br>
                • <code>days_to_expiry >= 7</code> - Days to expiry at least 7<br>
                • <code>premium_ratio_pct > 20</code> - Premium ratio greater than 20%<br>
                • <code>s_day_prem > 1000</code> - Short daily premium greater than $1000<br>
                • <code>trade_quality > 0</code> - Trade quality score positive<br>
                • <code>bid > 0</code> - Short leg bid price greater than 0<br>
                • <code>ask < 5</code> - Short leg ask price less than $5<br>
                • <code>l_bid > 10</code> - Long leg bid price greater than $10<br>
                <strong>Spread Filters:</strong><br>
                • <code>spread < 0.1</code> - Short leg bid/ask spread less than $0.10 (absolute value)<br>
                • <code>spread < 10%</code> - Short leg spread less than 10% of option premium (percentage-based)<br>
                • <code>l_spread < 0.1</code> - Long leg bid/ask spread less than $0.10 (absolute value)<br>
                • <code>l_spread < 10%</code> - Long leg spread less than 10% of option premium (percentage-based)<br>
                <strong>💡 Spread Filter Tip:</strong> Use percentage-based spread filters (e.g., <code>spread < 20%</code>) to filter by spread relative to option premium. This helps find liquid options where the bid/ask spread is a small percentage of the premium value.<br>
                <strong>Advanced Filters:</strong><br>
                • <code>num_contracts > volume</code> - Field-to-field comparison<br>
                • <code>curr_price*1.05 < strike_price</code> - Mathematical expression (5% above current price less than strike)<br>
                • <code>strike_price*0.95 > curr_price</code> - Mathematical expression (strike 5% below current)<br>
                • <code>opt_prem. / l_prem > 0.2</code> - Premium ratio calculation<br>
                <strong>Operators:</strong> <code>&gt;</code> <code>&gt;=</code> <code>&lt;</code> <code>&lt;=</code> <code>==</code> <code>!=</code> <code>exists</code> <code>not_exists</code><br>
                <strong>Math Operations:</strong> Use <code>+</code> <code>-</code> <code>*</code> <code>/</code> in expressions (e.g., <code>field*1.05</code>, <code>field+100</code>)<br>
                <strong>💡 Tip:</strong> When the filter section is expanded, column headers show their filterable field names. Filters are automatically saved in the URL - share the URL to share your filtered view!
            </div>
        </div>
""")
            # Loading indicator
            html_parts.append(f"""        <div class="loading-indicator" id="{prefix}loadingIndicator">
            <div class="loading-spinner"></div>
            <div class="loading-text">Loading data...</div>
        </div>
""")
            html_parts.append(f'        <div class="table-wrapper hide-hidden" id="{prefix}tableWrapper">\n')
            html_parts.append(f'            <table id="{prefix}resultsTable">\n')
            html_parts.append("                <thead>\n")
            html_parts.extend(build_table_headers(df_display, column_groups, column_to_group, hidden_set, always_hidden_set, prefix))
            html_parts.append("                </thead>\n")
            html_parts.append("                <tbody>\n")
            html_parts.append('                    <tr><td colspan="100%" style="text-align: center; padding: 20px;">Loading data...</td></tr>\n')
            html_parts.append("                </tbody>\n")
            html_parts.append("            </table>\n")
            html_parts.append("        </div>\n")
            return ''.join(html_parts)
        else:
            # Create minimal table structure if no data available
            return f"""        <div style="margin-bottom: 15px; display: flex; justify-content: space-between; gap: 10px; align-items: center;">
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
                <strong>📋 All Filterable Fields:</strong><br>
                <strong>Stock Info:</strong> <code>ticker</code>, <code>current_price</code> (or <code>curr_price</code>), <code>pe_ratio</code>, <code>market_cap_b</code>, <code>price_change_pct</code>, <code>price_with_change</code><br>
                <strong>Strike Prices:</strong> <code>strike_price</code>, <code>l_strike</code> (long strike)<br>
                <strong>Expiration Dates:</strong> <code>expiration_date</code>, <code>l_expiration_date</code> (long expiration), <code>days_to_expiry</code>, <code>l_days_to_expiry</code><br>
                <strong>Option Premiums:</strong> <code>opt_prem.</code> (or <code>option_premium</code>), <code>l_prem</code> (or <code>l_opt_prem</code>), <code>premium_ratio_pct</code>, <code>s_prem_tot</code>, <code>l_prem_tot</code>, <code>net_premium</code><br>
                <strong>Bid/Ask:</strong> <code>bid:ask</code>, <code>l_bid:ask</code> (long bid/ask)<br>
                <strong>Greeks:</strong> <code>delta</code>, <code>l_delta</code> (long delta), <code>theta</code>, <code>l_theta</code> (long theta)<br>
                <strong>Daily Premiums:</strong> <code>s_day_prem</code>, <code>net_daily_premi</code> (or <code>net_daily_premium</code>), <code>net_daily_premium_after_spread</code><br>
                <strong>Trading:</strong> <code>volume</code>, <code>num_contracts</code>, <code>option_ticker</code>, <code>l_option_ticker</code><br>
                <strong>Spread Analysis:</strong> <code>spread</code> (virtual field - short leg spread), <code>l_spread</code> (virtual field - long leg spread), <code>spread_slippage</code>, <code>spread_impact_pct</code><br>
                <strong>Bid/Ask Virtual Fields:</strong> <code>bid</code> (virtual - extracts bid from bid:ask), <code>ask</code> (virtual - extracts ask from bid:ask), <code>l_bid</code> (virtual - extracts bid from l_bid:ask), <code>l_ask</code> (virtual - extracts ask from l_bid:ask)<br>
                <strong>Quality Metrics:</strong> <code>trade_quality</code>, <code>liquidity_score</code>, <code>assignment_risk</code><br>
                <strong>Other:</strong> <code>option_type</code>, <code>buy_cost</code>, <code>price_above_curr</code>, <code>premium_diff</code>, <code>latest_option_writets</code><br>
                <br>
                <strong>Filter Examples:</strong><br>
                • <code>pe_ratio > 20</code> - P/E ratio greater than 20<br>
                • <code>market_cap_b < 3.5</code> - Market cap less than 3.5B<br>
                • <code>volume exists</code> - Volume data exists<br>
                • <code>net_daily_premium > 100</code> - Net daily premium greater than 100<br>
                • <code>delta < 0.5</code> - Delta less than 0.5<br>
                • <code>days_to_expiry >= 7</code> - Days to expiry at least 7<br>
                • <code>premium_ratio_pct > 20</code> - Premium ratio greater than 20%<br>
                • <code>s_day_prem > 1000</code> - Short daily premium greater than $1000<br>
                • <code>trade_quality > 0</code> - Trade quality score positive<br>
                • <code>bid > 0</code> - Short leg bid price greater than 0<br>
                • <code>ask < 5</code> - Short leg ask price less than $5<br>
                • <code>l_bid > 10</code> - Long leg bid price greater than $10<br>
                <strong>Spread Filters:</strong><br>
                • <code>spread < 0.1</code> - Short leg bid/ask spread less than $0.10 (absolute value)<br>
                • <code>spread < 10%</code> - Short leg spread less than 10% of option premium (percentage-based)<br>
                • <code>l_spread < 0.1</code> - Long leg bid/ask spread less than $0.10 (absolute value)<br>
                • <code>l_spread < 10%</code> - Long leg spread less than 10% of option premium (percentage-based)<br>
                <strong>💡 Spread Filter Tip:</strong> Use percentage-based spread filters (e.g., <code>spread < 20%</code>) to filter by spread relative to option premium. This helps find liquid options where the bid/ask spread is a small percentage of the premium value.<br>
                <strong>Advanced Filters:</strong><br>
                • <code>num_contracts > volume</code> - Field-to-field comparison<br>
                • <code>curr_price*1.05 < strike_price</code> - Mathematical expression (5% above current price less than strike)<br>
                • <code>strike_price*0.95 > curr_price</code> - Mathematical expression (strike 5% below current)<br>
                • <code>opt_prem. / l_prem > 0.2</code> - Premium ratio calculation<br>
                <strong>Operators:</strong> <code>&gt;</code> <code>&gt;=</code> <code>&lt;</code> <code>&lt;=</code> <code>==</code> <code>!=</code> <code>exists</code> <code>not_exists</code><br>
                <strong>Math Operations:</strong> Use <code>+</code> <code>-</code> <code>*</code> <code>/</code> in expressions (e.g., <code>field*1.05</code>, <code>field+100</code>)<br>
                <strong>💡 Tip:</strong> When the filter section is expanded, column headers show their filterable field names. Filters are automatically saved in the URL - share the URL to share your filtered view!
            </div>
        </div>
        <div class="table-wrapper hide-hidden" id="{prefix}tableWrapper">
            <table id="{prefix}resultsTable">
                <thead>
                    <tr class="column-header-row">
                        <th>Loading...</th>
                    </tr>
                </thead>
                <tbody>
                    <tr><td colspan="100%" style="text-align: center; padding: 20px;">Loading data...</td></tr>
                </tbody>
            </table>
        </div>
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
                <strong>📋 All Filterable Fields:</strong><br>
                <strong>Stock Info:</strong> <code>ticker</code>, <code>current_price</code> (or <code>curr_price</code>), <code>pe_ratio</code>, <code>market_cap_b</code>, <code>price_change_pct</code>, <code>price_with_change</code><br>
                <strong>Strike Prices:</strong> <code>strike_price</code>, <code>l_strike</code> (long strike)<br>
                <strong>Expiration Dates:</strong> <code>expiration_date</code>, <code>l_expiration_date</code> (long expiration), <code>days_to_expiry</code>, <code>l_days_to_expiry</code><br>
                <strong>Option Premiums:</strong> <code>opt_prem.</code> (or <code>option_premium</code>), <code>l_prem</code> (or <code>l_opt_prem</code>), <code>premium_ratio_pct</code>, <code>s_prem_tot</code>, <code>l_prem_tot</code>, <code>net_premium</code><br>
                <strong>Bid/Ask:</strong> <code>bid:ask</code>, <code>l_bid:ask</code> (long bid/ask)<br>
                <strong>Greeks:</strong> <code>delta</code>, <code>l_delta</code> (long delta), <code>theta</code>, <code>l_theta</code> (long theta)<br>
                <strong>Daily Premiums:</strong> <code>s_day_prem</code>, <code>net_daily_premi</code> (or <code>net_daily_premium</code>), <code>net_daily_premium_after_spread</code><br>
                <strong>Trading:</strong> <code>volume</code>, <code>num_contracts</code>, <code>option_ticker</code>, <code>l_option_ticker</code><br>
                <strong>Spread Analysis:</strong> <code>spread</code> (virtual field - short leg spread), <code>l_spread</code> (virtual field - long leg spread), <code>spread_slippage</code>, <code>spread_impact_pct</code><br>
                <strong>Bid/Ask Virtual Fields:</strong> <code>bid</code> (virtual - extracts bid from bid:ask), <code>ask</code> (virtual - extracts ask from bid:ask), <code>l_bid</code> (virtual - extracts bid from l_bid:ask), <code>l_ask</code> (virtual - extracts ask from l_bid:ask)<br>
                <strong>Quality Metrics:</strong> <code>trade_quality</code>, <code>liquidity_score</code>, <code>assignment_risk</code><br>
                <strong>Other:</strong> <code>option_type</code>, <code>buy_cost</code>, <code>price_above_curr</code>, <code>premium_diff</code>, <code>latest_option_writets</code><br>
                <br>
                <strong>Filter Examples:</strong><br>
                • <code>pe_ratio > 20</code> - P/E ratio greater than 20<br>
                • <code>market_cap_b < 3.5</code> - Market cap less than 3.5B<br>
                • <code>volume exists</code> - Volume data exists<br>
                • <code>net_daily_premium > 100</code> - Net daily premium greater than 100<br>
                • <code>delta < 0.5</code> - Delta less than 0.5<br>
                • <code>days_to_expiry >= 7</code> - Days to expiry at least 7<br>
                • <code>premium_ratio_pct > 20</code> - Premium ratio greater than 20%<br>
                • <code>s_day_prem > 1000</code> - Short daily premium greater than $1000<br>
                • <code>trade_quality > 0</code> - Trade quality score positive<br>
                • <code>bid > 0</code> - Short leg bid price greater than 0<br>
                • <code>ask < 5</code> - Short leg ask price less than $5<br>
                • <code>l_bid > 10</code> - Long leg bid price greater than $10<br>
                <strong>Spread Filters:</strong><br>
                • <code>spread < 0.1</code> - Short leg bid/ask spread less than $0.10 (absolute value)<br>
                • <code>spread < 10%</code> - Short leg spread less than 10% of option premium (percentage-based)<br>
                • <code>l_spread < 0.1</code> - Long leg bid/ask spread less than $0.10 (absolute value)<br>
                • <code>l_spread < 10%</code> - Long leg spread less than 10% of option premium (percentage-based)<br>
                <strong>💡 Spread Filter Tip:</strong> Use percentage-based spread filters (e.g., <code>spread < 20%</code>) to filter by spread relative to option premium. This helps find liquid options where the bid/ask spread is a small percentage of the premium value.<br>
                <strong>Advanced Filters:</strong><br>
                • <code>num_contracts > volume</code> - Field-to-field comparison<br>
                • <code>curr_price*1.05 < strike_price</code> - Mathematical expression (5% above current price less than strike)<br>
                • <code>strike_price*0.95 > curr_price</code> - Mathematical expression (strike 5% below current)<br>
                • <code>opt_prem. / l_prem > 0.2</code> - Premium ratio calculation<br>
                <strong>Operators:</strong> <code>&gt;</code> <code>&gt;=</code> <code>&lt;</code> <code>&lt;=</code> <code>==</code> <code>!=</code> <code>exists</code> <code>not_exists</code><br>
                <strong>Math Operations:</strong> Use <code>+</code> <code>-</code> <code>*</code> <code>/</code> in expressions (e.g., <code>field*1.05</code>, <code>field+100</code>)<br>
                <strong>💡 Tip:</strong> When the filter section is expanded, column headers show their filterable field names. Filters are automatically saved in the URL - share the URL to share your filtered view!
            </div>
        </div>
""")
    
    # Loading indicator
    html_parts.append(f"""        <div class="loading-indicator" id="{prefix}loadingIndicator">
            <div class="loading-spinner"></div>
            <div class="loading-text">Loading data...</div>
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
            group_info = column_groups[group_name]
            short_cols = group_info.get('short', [])
            long_cols = group_info.get('long', [])
            extras = group_info.get('extras', [])
            # Use the first short as the anchor for the header
            if col in short_cols:
                display_name = COLUMN_GROUPS[group_name]['display_name']
                colspan = len(short_cols) + len(long_cols) + len(extras)
                html_parts.append(f'                        <th class="group-header" colspan="{colspan}">{html_escape.escape(display_name)}</th>\n')
                for c in short_cols + long_cols + extras:
                    processed_cols.add(c)
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
            group_info = column_groups[group_name]
            short_cols = group_info.get('short', [])
            long_cols = group_info.get('long', [])
            # Label only the primary short/long columns; extras (like premium_ratio_pct) get no sub-label
            if col in short_cols:
                short_long_label = 'Short'
            elif col in long_cols:
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

