"""
Mobile card HTML generation.
"""

import pandas as pd
import html as html_escape
from typing import List
from .config import PRIMARY_CARD_COLUMNS, PRIMARY_COLUMN_LABELS
from .data_processor import normalize_col_name


def build_cards_html(
    df_display: pd.DataFrame,
    df_raw: pd.DataFrame,
    prefix: str
) -> str:
    """Build mobile card HTML.
    
    Args:
        df_display: Formatted DataFrame for display
        df_raw: Raw DataFrame for filtering
        prefix: Tab prefix ('calls' or 'puts')
        
    Returns:
        Cards HTML as string
    """
    if df_display.empty:
        return '<div style="padding: 20px; text-align: center;"><p>No data available.</p></div>'
    
    html_parts = []
    html_parts.append(f'        <div class="card-wrapper" id="{prefix}cardWrapper">\n')
    
    for row_idx, row in df_display.iterrows():
        html_parts.append(build_single_card(row, row_idx, df_raw, prefix))
    
    html_parts.append("        </div>\n")
    
    return ''.join(html_parts)


def build_single_card(
    row: pd.Series,
    row_idx: int,
    df_raw: pd.DataFrame,
    prefix: str
) -> str:
    """Build a single card HTML.
    
    Args:
        row: Row data from DataFrame
        row_idx: Row index
        df_raw: Raw DataFrame for data attributes
        prefix: Tab prefix
        
    Returns:
        Single card HTML as string
    """
    html_parts = []
    
    # Get ticker
    ticker = str(row.get('ticker', 'N/A')) if pd.notna(row.get('ticker')) else 'N/A'
    
    # Get current price
    current_price = ''
    for price_col in ['current_price', 'curr_price', 'cur_price']:
        if price_col in row.index and pd.notna(row[price_col]) and str(row[price_col]).strip():
            current_price = str(row[price_col])
            break
    
    # Get price change
    change_pct = ''
    change_class = ''
    for change_col in ['change_pct', 'price_with_change', 'price_with_chan']:
        if change_col in row.index and pd.notna(row[change_col]) and str(row[change_col]).strip():
            change_pct = str(row[change_col])
            change_str = change_pct
            if '+$' in change_str or '(+' in change_str:
                change_class = 'positive'
            elif '-$' in change_str or '(-' in change_str:
                change_class = 'negative'
            break
    
    # Build data attributes for filtering
    data_attrs = []
    if row_idx in df_raw.index:
        for col in ['ticker', 'current_price', 'strike_price', 'option_premium', 'daily_premium', 'net_daily_premium', 'volume', 'delta', 'theta']:
            if col in df_raw.columns:
                raw_val = df_raw.loc[row_idx, col]
                if pd.notna(raw_val):
                    try:
                        if isinstance(raw_val, (int, float)):
                            data_attrs.append(f'data-{normalize_col_name(col)}="{raw_val}"')
                        else:
                            val_str = str(raw_val)
                            import re
                            match = re.search(r'-?\d+\.?\d*', val_str)
                            if match:
                                data_attrs.append(f'data-{normalize_col_name(col)}="{match.group()}"')
                    except:
                        pass
    
    data_attrs_str = ' ' + ' '.join(data_attrs) if data_attrs else ''
    
    # Card HTML
    html_parts.append(f'            <div class="data-card" data-row-index="{row_idx}"{data_attrs_str}>\n')
    
    # Card header
    html_parts.append('                <div class="card-header">\n')
    html_parts.append('                    <div class="card-header-main">\n')
    # Make ticker a link
    if ticker and ticker != 'N/A':
        ticker_link = f'/stock_info/{html_escape.escape(ticker)}'
        html_parts.append(f'                        <div class="card-ticker"><a href="{ticker_link}" target="_blank" style="color: inherit; text-decoration: none; font-weight: 500;">{html_escape.escape(ticker)}</a></div>\n')
    else:
        html_parts.append(f'                        <div class="card-ticker">{html_escape.escape(ticker)}</div>\n')
    html_parts.append('                        <div class="card-price">\n')
    
    if current_price:
        price_display = current_price if current_price.startswith('$') else f'${current_price}'
        html_parts.append(f'                            <span class="card-price-value">{html_escape.escape(price_display)}</span>\n')
    
    if change_pct:
        html_parts.append(f'                            <span class="card-price-change {change_class}">{html_escape.escape(change_pct)}</span>\n')
    
    html_parts.append('                        </div>\n')
    html_parts.append('                    </div>\n')
    html_parts.append('                </div>\n')
    
    # Card body
    html_parts.append('                <div class="card-body">\n')
    
    # Primary metrics
    html_parts.append('                    <div class="card-primary">\n')
    shown_primary = set()
    
    for col_key in PRIMARY_CARD_COLUMNS:
        col_name = find_column_by_normalized(row, col_key)
        if col_name and col_name in row.index:
            val = row[col_name]
            if pd.notna(val) and str(val).strip() and col_name not in shown_primary:
                label = PRIMARY_COLUMN_LABELS.get(col_key, col_key.replace('_', ' ').title())
                html_parts.append('                        <div class="card-primary-item">\n')
                html_parts.append(f'                            <div class="card-primary-label">{html_escape.escape(label)}</div>\n')
                html_parts.append(f'                            <div class="card-primary-value">{html_escape.escape(str(val))}</div>\n')
                html_parts.append('                        </div>\n')
                shown_primary.add(col_name)
    
    html_parts.append('                    </div>\n')
    
    # Expandable details
    html_parts.append(f'                    <div class="card-details" id="{prefix}cardDetails_{row_idx}" style="display: none;">\n')
    
    # Categorize and add details
    details_html = build_card_details(row, shown_primary)
    html_parts.append(details_html)
    
    html_parts.append('                    </div>\n')
    
    # Toggle button
    html_parts.append(f'                    <button class="card-toggle" onclick="toggleCardDetails(\'{prefix}\', {row_idx})" id="{prefix}cardToggle_{row_idx}">\n')
    html_parts.append('                        <span>Show More Details</span>\n')
    html_parts.append('                        <span class="card-toggle-icon">▼</span>\n')
    html_parts.append('                    </button>\n')
    
    html_parts.append('                </div>\n')
    html_parts.append('            </div>\n')
    
    return ''.join(html_parts)


def find_column_by_normalized(row: pd.Series, target_normalized: str) -> str:
    """Find column in row by normalized name.
    
    Args:
        row: Row Series
        target_normalized: Normalized column name to find
        
    Returns:
        Column name if found, None otherwise
    """
    for col in row.index:
        if normalize_col_name(col) == target_normalized:
            return col
    return None


def build_card_details(row: pd.Series, shown_primary: set) -> str:
    """Build expandable card details section.
    
    Args:
        row: Row data
        shown_primary: Set of columns already shown in primary section
        
    Returns:
        Details HTML as string
    """
    html_parts = []
    
    # Categorize columns
    option_cols = []
    greeks_cols = []
    premium_cols = []
    other_cols = []
    
    for col in row.index:
        normalized = normalize_col_name(col)
        
        # Skip already shown or special columns
        if normalized in ['ticker', 'current_price', 'curr_price', 'change_pct', 'price_with_change']:
            continue
        if col in shown_primary:
            continue
        
        val = row[col]
        if pd.isna(val) or str(val).strip() == '':
            continue
        
        col_label = col.replace('_', ' ').title()
        
        # Categorize
        if any(x in normalized for x in ['strike', 'expiration', 'expiry', 'option_ticker', 'bid', 'ask']):
            option_cols.append((col, col_label, val))
        elif any(x in normalized for x in ['delta', 'theta', 'gamma', 'vega', 'iv', 'implied']):
            greeks_cols.append((col, col_label, val))
        elif any(x in normalized for x in ['premium', 'prem', 'net', 'total', 'daily']):
            premium_cols.append((col, col_label, val))
        else:
            other_cols.append((col, col_label, val))
    
    # Render sections
    if option_cols:
        html_parts.append('                        <div class="card-section">\n')
        html_parts.append('                            <div class="card-section-title">Option Details</div>\n')
        for col, col_label, val in option_cols:
            html_parts.append('                            <div class="card-row">\n')
            html_parts.append(f'                                <span class="card-label">{html_escape.escape(col_label)}</span>\n')
            html_parts.append(f'                                <span class="card-value">{html_escape.escape(str(val))}</span>\n')
            html_parts.append('                            </div>\n')
        html_parts.append('                        </div>\n')
    
    if greeks_cols:
        html_parts.append('                        <div class="card-section">\n')
        html_parts.append('                            <div class="card-section-title">Greeks</div>\n')
        for col, col_label, val in greeks_cols:
            html_parts.append('                            <div class="card-row">\n')
            html_parts.append(f'                                <span class="card-label">{html_escape.escape(col_label)}</span>\n')
            html_parts.append(f'                                <span class="card-value">{html_escape.escape(str(val))}</span>\n')
            html_parts.append('                            </div>\n')
        html_parts.append('                        </div>\n')
    
    if premium_cols:
        html_parts.append('                        <div class="card-section">\n')
        html_parts.append('                            <div class="card-section-title">Premium & Returns</div>\n')
        for col, col_label, val in premium_cols:
            html_parts.append('                            <div class="card-row">\n')
            html_parts.append(f'                                <span class="card-label">{html_escape.escape(col_label)}</span>\n')
            html_parts.append(f'                                <span class="card-value">{html_escape.escape(str(val))}</span>\n')
            html_parts.append('                            </div>\n')
        html_parts.append('                        </div>\n')
    
    if other_cols:
        html_parts.append('                        <div class="card-section">\n')
        html_parts.append('                            <div class="card-section-title">Other</div>\n')
        for col, col_label, val in other_cols:
            html_parts.append('                            <div class="card-row">\n')
            html_parts.append(f'                                <span class="card-label">{html_escape.escape(col_label)}</span>\n')
            html_parts.append(f'                                <span class="card-value">{html_escape.escape(str(val))}</span>\n')
            html_parts.append('                            </div>\n')
        html_parts.append('                        </div>\n')
    
    return ''.join(html_parts)

