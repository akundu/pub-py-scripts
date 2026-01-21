"""
HTML generation utilities for creating stock information pages.

This module provides functions to generate Yahoo Finance-like HTML pages for stock data,
including price charts, financial metrics, options data, and news.
"""

import json
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Dict, Any

# Import serialization utilities
from .serializers import dataframe_to_json_records

logger = logging.getLogger("db_server_logger")

def format_options_html(options_data: Dict[str, Any], current_price: float = None) -> str:
    """Format options data as HTML table with calls and puts side-by-side, bid/ask combined with midpoint."""
    if not options_data or not options_data.get('success', False):
        return '<p>No options data available</p>'
    
    contracts = options_data.get('data', {}).get('contracts', [])
    if not contracts:
        return '<p>No options contracts found</p>'
    
    # Helper function to calculate background color based on moneyness
    def get_row_bg_color(strike, current_price, option_type):
        """Calculate background color based on how far strike is from current price."""
        if not current_price:
            return ''
        
        pct_diff = abs(strike - current_price) / current_price * 100
        
        # Determine if in-the-money (ITM)
        if option_type == 'call':
            itm = strike < current_price
        else:  # put
            itm = strike > current_price
        
        # Color intensity based on distance from ATM
        if pct_diff < 2:  # Very close to ATM
            alpha = 0.3
        elif pct_diff < 5:
            alpha = 0.2
        elif pct_diff < 10:
            alpha = 0.1
        else:
            alpha = 0.05
        
        # ITM: yellow tint, OTM: light gray
        if itm:
            return f'background-color: rgba(255, 235, 59, {alpha});'  # Yellow for ITM
        else:
            return f'background-color: rgba(200, 200, 200, {alpha});'  # Gray for OTM
    
    # Group by expiration date
    by_expiry = {}
    for contract in contracts:
        exp = contract.get('expiration', 'Unknown')
        if exp not in by_expiry:
            by_expiry[exp] = []
        by_expiry[exp].append(contract)
    
    html_parts = []
    for exp_date in sorted(by_expiry.keys())[:10]:  # Show first 10 expirations
        contracts_list = by_expiry[exp_date]
        
        # Group by strike price and option type
        by_strike = {}
        for contract in contracts_list:
            strike = contract.get('strike', 0)
            if not isinstance(strike, (int, float)):
                continue
            # Support both 'type' and 'option_type' keys
            option_type = str(contract.get('option_type', contract.get('type', ''))).lower()
            if strike not in by_strike:
                by_strike[strike] = {'call': None, 'put': None}
            by_strike[strike][option_type] = contract
        
        # Sort strikes high to low
        sorted_strikes = sorted(by_strike.keys(), reverse=True)
        
        html_parts.append(f'<h3 style="margin-top: 20px; color: #667eea;">Expiration: {exp_date}</h3>')
        html_parts.append('<table class="data-table" style="width: 100%; margin-bottom: 20px; font-size: 12px; border-collapse: collapse;">')
        
        # Header with calls on left, puts on right
        html_parts.append('''
        <tr>
            <th colspan="6" style="background-color: #4CAF50; color: white; padding: 8px; font-weight: bold; font-size: 14px;">CALLS</th>
            <th rowspan="2" style="background-color: #667eea; color: white; padding: 8px; font-weight: bold; font-size: 14px;">Strike</th>
            <th colspan="6" style="background-color: #f44336; color: white; padding: 8px; font-weight: bold; font-size: 14px;">PUTS</th>
        </tr>
        <tr>
            <th style="padding: 6px; background-color: #2e7d32; color: white; font-weight: bold; font-size: 12px;">Bid/Ask<br>Spread</th>
            <th style="padding: 6px; background-color: #2e7d32; color: white; font-weight: bold; font-size: 12px;">Mid</th>
            <th style="padding: 6px; background-color: #2e7d32; color: white; font-weight: bold; font-size: 12px;">Vol</th>
            <th style="padding: 6px; background-color: #2e7d32; color: white; font-weight: bold; font-size: 12px;">IV</th>
            <th style="padding: 6px; background-color: #2e7d32; color: white; font-weight: bold; font-size: 12px;">Delta<br>(Δ)</th>
            <th style="padding: 6px; background-color: #2e7d32; color: white; font-weight: bold; font-size: 12px;">Theta<br>(Θ)</th>
            <th style="padding: 6px; background-color: #c62828; color: white; font-weight: bold; font-size: 12px;">Bid/Ask<br>Spread</th>
            <th style="padding: 6px; background-color: #c62828; color: white; font-weight: bold; font-size: 12px;">Mid</th>
            <th style="padding: 6px; background-color: #c62828; color: white; font-weight: bold; font-size: 12px;">Vol</th>
            <th style="padding: 6px; background-color: #c62828; color: white; font-weight: bold; font-size: 12px;">IV</th>
            <th style="padding: 6px; background-color: #c62828; color: white; font-weight: bold; font-size: 12px;">Delta<br>(Δ)</th>
            <th style="padding: 6px; background-color: #c62828; color: white; font-weight: bold; font-size: 12px;">Theta<br>(Θ)</th>
        </tr>
        ''')
        
        # Show up to 30 strikes per expiration
        for strike in sorted_strikes[:30]:
            call = by_strike[strike]['call']
            put = by_strike[strike]['put']
            
            # Calculate row background color
            call_bg = get_row_bg_color(strike, current_price, 'call') if current_price else ''
            put_bg = get_row_bg_color(strike, current_price, 'put') if current_price else ''
            
            html_parts.append('<tr>')
            
            # CALL data
            if call:
                bid = call.get('bid')
                ask = call.get('ask')
                last = call.get('last')
                volume = call.get('volume', 'N/A')
                oi = call.get('open_interest', 'N/A')
                iv = call.get('implied_volatility')
                delta = call.get('delta')
                theta = call.get('theta')
                
                # Bid/Ask/Spread column - bid and ask on same line, spread below
                if isinstance(bid, (int, float)) and isinstance(ask, (int, float)) and bid > 0 and ask > 0:
                    spread = ask - bid
                    html_parts.append(f'<td style="padding: 4px; {call_bg}">${bid:.2f} / ${ask:.2f}<br><strong>${spread:.2f}</strong></td>')
                elif isinstance(bid, (int, float)):
                    html_parts.append(f'<td style="padding: 4px; {call_bg}">${bid:.2f} / -<br>-</td>')
                elif isinstance(ask, (int, float)):
                    html_parts.append(f'<td style="padding: 4px; {call_bg}">- / ${ask:.2f}<br>-</td>')
                else:
                    html_parts.append(f'<td style="padding: 4px; {call_bg}">-</td>')
                
                # Mid column
                if isinstance(bid, (int, float)) and isinstance(ask, (int, float)) and bid > 0 and ask > 0:
                    mid = (bid + ask) / 2
                    html_parts.append(f'<td style="padding: 4px; {call_bg}"><strong>${mid:.2f}</strong></td>')
                else:
                    html_parts.append(f'<td style="padding: 4px; {call_bg}">-</td>')
                
                html_parts.append(f'<td style="padding: 4px; {call_bg}">{volume}</td>')
                # Format IV as percentage (e.g., 0.25 -> 25.00%)
                if isinstance(iv, (int, float)) and iv is not None:
                    iv_pct = iv * 100
                    html_parts.append(f'<td style="padding: 4px; {call_bg}">{iv_pct:.2f}%</td>')
                else:
                    html_parts.append(f'<td style="padding: 4px; {call_bg}">-</td>')
                
                # Delta (separate column)
                html_parts.append(f'<td style="padding: 4px; {call_bg}"><strong>{delta:.3f}</strong></td>' if isinstance(delta, (int, float)) else f'<td style="padding: 4px; {call_bg}">-</td>')
                
                # Theta (separate column)
                html_parts.append(f'<td style="padding: 4px; {call_bg}">{theta:.3f}</td>' if isinstance(theta, (int, float)) else f'<td style="padding: 4px; {call_bg}">-</td>')
            else:
                html_parts.append('<td style="padding: 4px;">-</td>' * 6)
            
            # Strike price (center) - highlight if near current price
            strike_style = 'background-color: #667eea; color: white; font-weight: bold; padding: 6px;'
            if current_price and abs(strike - current_price) / current_price < 0.02:  # Within 2% of current
                strike_style = 'background-color: #ff9800; color: white; font-weight: bold; padding: 6px; border: 2px solid #f57c00;'
            html_parts.append(f'<td style="{strike_style}">${strike:.2f}</td>')
            
            # PUT data
            if put:
                bid = put.get('bid')
                ask = put.get('ask')
                last = put.get('last')
                volume = put.get('volume', 'N/A')
                oi = put.get('open_interest', 'N/A')
                iv = put.get('implied_volatility')
                delta = put.get('delta')
                theta = put.get('theta')
                
                # Bid/Ask/Spread column - bid and ask on same line, spread below
                if isinstance(bid, (int, float)) and isinstance(ask, (int, float)) and bid > 0 and ask > 0:
                    spread = ask - bid
                    html_parts.append(f'<td style="padding: 4px; {put_bg}">${bid:.2f} / ${ask:.2f}<br><strong>${spread:.2f}</strong></td>')
                elif isinstance(bid, (int, float)):
                    html_parts.append(f'<td style="padding: 4px; {put_bg}">${bid:.2f} / -<br>-</td>')
                elif isinstance(ask, (int, float)):
                    html_parts.append(f'<td style="padding: 4px; {put_bg}">- / ${ask:.2f}<br>-</td>')
                else:
                    html_parts.append(f'<td style="padding: 4px; {put_bg}">-</td>')
                
                # Mid column
                if isinstance(bid, (int, float)) and isinstance(ask, (int, float)) and bid > 0 and ask > 0:
                    mid = (bid + ask) / 2
                    html_parts.append(f'<td style="padding: 4px; {put_bg}"><strong>${mid:.2f}</strong></td>')
                else:
                    html_parts.append(f'<td style="padding: 4px; {put_bg}">-</td>')
                
                html_parts.append(f'<td style="padding: 4px; {put_bg}">{volume}</td>')
                # Format IV as percentage (e.g., 0.25 -> 25.00%)
                if isinstance(iv, (int, float)) and iv is not None:
                    iv_pct = iv * 100
                    html_parts.append(f'<td style="padding: 4px; {put_bg}">{iv_pct:.2f}%</td>')
                else:
                    html_parts.append(f'<td style="padding: 4px; {put_bg}">-</td>')
                
                # Delta (separate column)
                html_parts.append(f'<td style="padding: 4px; {put_bg}"><strong>{delta:.3f}</strong></td>' if isinstance(delta, (int, float)) else f'<td style="padding: 4px; {put_bg}">-</td>')
                
                # Theta (separate column)
                html_parts.append(f'<td style="padding: 4px; {put_bg}">{theta:.3f}</td>' if isinstance(theta, (int, float)) else f'<td style="padding: 4px; {put_bg}">-</td>')
            else:
                html_parts.append('<td style="padding: 4px;">-</td>' * 6)
            
            html_parts.append('</tr>')
        
        html_parts.append('</table>')
    
    return ''.join(html_parts)


def generate_stock_info_html(symbol: str, data: Dict[str, Any], earnings_date: str = None) -> str:
    """Generate Yahoo Finance-like HTML page for stock information."""
    import json
    from datetime import datetime, timedelta
    import pandas as pd
    
    def format_value(val):
        """Format a value for display."""
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return 'N/A'
        if isinstance(val, (int, float)):
            if val >= 1e9:
                return f"${val/1e9:.2f}B"
            elif val >= 1e6:
                return f"${val/1e6:.2f}M"
            elif val >= 1e3:
                return f"${val/1e3:.2f}K"
            else:
                return f"{val:.2f}"
        return str(val)
    
    # Extract data
    price_info = data.get('price_info', {})
    financial_info = data.get('financial_info', {})
    options_info = data.get('options_info', {})
    iv_info = data.get('iv_info', {})
    news_info = data.get('news_info', {})
    
    # Get current price
    current_price_data = price_info.get('current_price', {})
    if isinstance(current_price_data, dict):
        current_price = current_price_data.get('price') or current_price_data.get('close') or current_price_data.get('last_price') or 'N/A'
        price_change = current_price_data.get('change', 0) or current_price_data.get('change_amount', 0)
        price_change_pct = current_price_data.get('change_percent', 0) or current_price_data.get('change_pct', 0)
        volume = current_price_data.get('volume') or current_price_data.get('size')
        
        # Get daily range (high/low for today)
        daily_range = current_price_data.get('daily_range')
        if daily_range and isinstance(daily_range, dict):
            daily_high = daily_range.get('high')
            daily_low = daily_range.get('low')
        else:
            daily_high = None
            daily_low = None
    else:
        # If current_price is a number directly
        current_price = current_price_data if isinstance(current_price_data, (int, float)) else 'N/A'
        price_change = 0
        price_change_pct = 0
        volume = None
        daily_high = None
        daily_low = None
    
    # Format price
    if isinstance(current_price, (int, float)):
        current_price_str = f"{current_price:.2f}"
    else:
        current_price_str = str(current_price)
    
    # Get price history for chart (daily) and merged series (realtime+hourly+daily)
    price_history = price_info.get('price_data', [])
    merged_price_series = price_info.get('merged_price_series')
    
    # Debug: Log price_history info
    if price_history is None:
        logger.warning(f"No price_history for {symbol} - price_data is None")
    elif hasattr(price_history, 'empty'):
        if price_history.empty:
            logger.warning(f"price_history is empty DataFrame for {symbol}")
        else:
            logger.info(f"price_history is DataFrame: rows={len(price_history)}, shape={price_history.shape}, columns={list(price_history.columns)}")
    elif isinstance(price_history, list):
        logger.info(f"price_history is list: length={len(price_history)}")
    else:
        logger.warning(f"price_history type: {type(price_history)} for {symbol}")
    
    # Get financial data - handle both dict and DataFrame
    financial_data = {}
    if financial_info:
        fin_data = financial_info.get('financial_data')
        if isinstance(fin_data, dict):
            financial_data = fin_data
        elif hasattr(fin_data, 'to_dict'):
            # It's a DataFrame - get the latest row
            if not fin_data.empty:
                financial_data = fin_data.iloc[-1].to_dict()
    
    # Calculate 52 week high/low from price history
    week_52_high = None
    week_52_low = None
    if price_history is not None:
        # Don't convert here - let the chart data extraction handle it
        # This avoids double conversion and ensures date field is preserved
        temp_price_history = price_history
        if hasattr(temp_price_history, 'to_dict'):
            # It's a DataFrame - convert it properly with date field
            df = temp_price_history.copy()
            # Check if 'date' column already exists
            if 'date' not in df.columns:
                # Always reset index to include it as a column
                if not df.index.empty:
                    is_range_index = isinstance(df.index, pd.RangeIndex)
                    if not is_range_index:
                        index_name = df.index.name if df.index.name else 'date'
                        df = df.reset_index()
                        if index_name in df.columns and index_name != 'date':
                            df = df.rename(columns={index_name: 'date'})
                        elif 'index' in df.columns:
                            df = df.rename(columns={'index': 'date'})
                        if 'date' not in df.columns and len(df.columns) > 0:
                            first_col = df.columns[0]
                            standard_cols = ['ticker', 'open', 'high', 'low', 'close', 'volume', 'ma_10', 'ma_50', 'ma_100', 'ma_200', 'ema_8', 'ema_21', 'ema_34', 'ema_55', 'ema_89', 'write_timestamp']
                            if first_col not in standard_cols:
                                df = df.rename(columns={first_col: 'date'})
            temp_price_history = dataframe_to_json_records(df)
        
        if isinstance(temp_price_history, list) and len(temp_price_history) > 0:
            # Get last 365 days of data
            prices = []
            for record in temp_price_history:
                if isinstance(record, dict):
                    close = record.get('close') or record.get('price')
                    if close:
                        try:
                            prices.append(float(close))
                        except (ValueError, TypeError):
                            pass
            if prices:
                week_52_high = max(prices)
                week_52_low = min(prices)
    
    # Process earnings date - filter out dividend/yield information
    earnings_date_display = None
    if earnings_date:
        # Filter out dividend/yield data - if it contains dividend or yield, it's not an earnings date
        if 'dividend' in earnings_date.lower() or 'yield' in earnings_date.lower():
            earnings_date_display = 'N/A'
        else:
            earnings_date_display = earnings_date.strip()
    
    # Format price change
    change_color = 'positive' if price_change >= 0 else 'negative'
    change_sign = '+' if price_change >= 0 else ''
    
    # Prepare chart data - prefer merged series if available
    chart_data = []
    chart_labels = []
    all_price_records = []
    
    # If we have a merged price series (from DB helper), use it for chart data.
    # merged_price_series is expected to be a list of dicts with at least:
    #   timestamp, close, source, is_daily_open, is_daily_close
    if isinstance(merged_price_series, list) and merged_price_series:
        logger.info(f"[HTML] Using merged_price_series for chart, records={len(merged_price_series)}")
        for rec in merged_price_series:
            if not isinstance(rec, dict):
                continue
            ts = rec.get('timestamp') or rec.get('date') or rec.get('datetime')
            close = rec.get('close') or rec.get('price') or rec.get('last_price')
            if not ts or close is None:
                continue
            try:
                close_val = float(close)
            except (TypeError, ValueError):
                continue
            # Normalize timestamp to ISO string for JS
            if not isinstance(ts, str):
                if hasattr(ts, 'isoformat'):
                    ts = ts.isoformat()
                else:
                    ts = str(ts)
            all_price_records.append({
                'timestamp': ts,
                'close': close_val,
                'source': rec.get('source', 'unknown'),
                'is_daily_open': bool(rec.get('is_daily_open', False)),
                'is_daily_close': bool(rec.get('is_daily_close', False)),
            })
    # Fallback: use daily price_history if merged series is not available
    elif price_history is not None:
        # Convert DataFrame to list of records if needed
        if hasattr(price_history, 'to_dict'):
            # It's a DataFrame - need to preserve the index (date) in records
            df = price_history.copy()
            logger.info(f"[HTML] Converting DataFrame for chart: index_type={type(df.index).__name__}, index_name={df.index.name}, columns={list(df.columns)}, has_date_col={'date' in df.columns}")
            # Check if 'date' column already exists (some databases return it as a column)
            if 'date' not in df.columns:
                # Always reset index to include it as a column (the index typically contains the date)
                # Only skip if it's a simple RangeIndex (0, 1, 2, ...)
                if not df.index.empty:
                    is_range_index = isinstance(df.index, pd.RangeIndex)
                    logger.info(f"[HTML] is_range_index={is_range_index}")
                    
                    # Reset index unless it's a RangeIndex
                    # Most databases set date as index, so we should reset it
                    if not is_range_index:
                        # Get the index name before resetting
                        index_name = df.index.name if df.index.name else 'date'
                        logger.info(f"[HTML] Resetting index with name: {index_name}")
                        df = df.reset_index()
                        logger.info(f"[HTML] After reset_index, columns: {list(df.columns)}")
                        # Rename the index column to 'date' for consistency
                        if index_name in df.columns and index_name != 'date':
                            df = df.rename(columns={index_name: 'date'})
                            logger.info(f"[HTML] Renamed {index_name} to 'date'")
                        elif 'index' in df.columns:
                            df = df.rename(columns={'index': 'date'})
                            logger.info(f"[HTML] Renamed 'index' to 'date'")
                        # If still no date column, check first column (might be datetime index converted)
                        if 'date' not in df.columns and len(df.columns) > 0:
                            first_col = df.columns[0]
                            standard_cols = ['ticker', 'open', 'high', 'low', 'close', 'volume', 'ma_10', 'ma_50', 'ma_100', 'ma_200', 'ema_8', 'ema_21', 'ema_34', 'ema_55', 'ema_89', 'write_timestamp']
                            if first_col not in standard_cols:
                                df = df.rename(columns={first_col: 'date'})
                                logger.info(f"[HTML] Renamed first column {first_col} to 'date'")
                    else:
                        # It's a RangeIndex - the date might be in a column already
                        # Check if there's a datetime column
                        logger.info(f"[HTML] RangeIndex detected, checking for date column")
                        for col in df.columns:
                            if col in ['date', 'datetime', 'timestamp'] or ('date' in col.lower() and col != 'update_date'):
                                df = df.rename(columns={col: 'date'})
                                logger.info(f"[HTML] Found and renamed {col} to 'date'")
                                break
            logger.info(f"[HTML] Final DataFrame columns before conversion: {list(df.columns)}, has_date={'date' in df.columns}")
            # Convert to records
            price_history = dataframe_to_json_records(df)
            logger.info(f"[HTML] After conversion, price_history type: {type(price_history)}, length: {len(price_history) if isinstance(price_history, list) else 'N/A'}")
            if isinstance(price_history, list) and len(price_history) > 0:
                logger.info(f"[HTML] First record keys: {list(price_history[0].keys())}, has_date: {'date' in price_history[0]}")
        
        # Now price_history should be a list
        if isinstance(price_history, list) and len(price_history) > 0:
            logger.info(f"[HTML] Processing {len(price_history)} records for chart")
            records_with_date = 0
            records_without_date = 0
            for record in price_history:
                if isinstance(record, dict):
                    # Try multiple possible date column names
                    date = (record.get('date') or 
                           record.get('timestamp') or 
                           record.get('datetime') or
                           record.get('time') or '')
                    # Try multiple possible price column names
                    close = (record.get('close') or 
                            record.get('price') or 
                            record.get('last_price') or 0)
                    if date:
                        records_with_date += 1
                    else:
                        records_without_date += 1
                        if records_without_date == 1:
                            logger.warning(f"[HTML] First record without date field. Keys: {list(record.keys())}")
                    if date and close:
                        try:
                            close_val = float(close)
                            # Ensure date is a string in ISO format for JavaScript
                            if not isinstance(date, str):
                                if hasattr(date, 'isoformat'):
                                    date = date.isoformat()
                                elif hasattr(date, 'strftime'):
                                    date = date.strftime('%Y-%m-%d')
                                else:
                                    date = str(date)
                            # Keep full timestamp string; JS will handle formatting
                            all_price_records.append({
                                'timestamp': date,
                                'close': close_val,
                                'source': 'daily',
                                'is_daily_open': False,
                                'is_daily_close': False,
                            })
                        except (ValueError, TypeError) as e:
                            logger.debug(f"Error processing price record: date={date}, close={close}, error={e}")
                            pass
            logger.info(f"[HTML] Chart data extraction: records_with_date={records_with_date}, records_without_date={records_without_date}, all_price_records={len(all_price_records)}")
    
    # Sort by timestamp and prepare chart data
    if all_price_records:
        # Sort by timestamp
        try:
            all_price_records.sort(key=lambda x: x['timestamp'])
        except Exception:
            pass
    
    # Convert to JSON for JavaScript - use all data, JavaScript will filter.
    # We keep simple arrays for backward compatibility, but also embed the full
    # merged records for richer styling (daily open/close markers).
    all_chart_data = [r['close'] for r in all_price_records]
    all_chart_labels = [r['timestamp'] for r in all_price_records]
    all_chart_data_json = json.dumps(all_chart_data)
    all_chart_labels_json = json.dumps(all_chart_labels)
    merged_series_json = json.dumps(all_price_records)
    
    # Get IV data
    iv_data = iv_info.get('iv_data', {}) if iv_info else {}
    
    # Get options data
    options_data = options_info.get('options_data', {}) if options_info else {}
    
    # Get news data
    news_data = news_info.get('news_data', {}) if news_info else {}
    # News data structure: {'articles': [...], 'count': N, 'fetched_at': ..., 'date_range': {...}}
    news_items = news_data.get('articles', []) if isinstance(news_data, dict) else []
    
    # Format news items HTML
    def format_news_item(item):
        title = item.get("title", "No title")
        published = item.get("published_utc", "")[:10] if item.get("published_utc") else ""
        description = item.get("description", "")
        article_url = item.get("article_url", "#")
        desc_html = f'<p>{description[:200]}...</p>' if description else ""
        return f'<li style="margin-bottom: 15px; padding: 10px; background: #f9fafb; border-radius: 4px;"><strong>{title}</strong><br><small>{published}</small><br>{desc_html}<a href="{article_url}" target="_blank" style="color: #667eea;">Read more</a></li>'
    
    news_html = ""
    if news_items:
        news_list_items = ''.join([format_news_item(item) for item in news_items[:10]])
        news_html = f'<ul style="list-style: none; padding: 0;">{news_list_items}</ul>'
    elif news_info and news_info.get('error'):
        news_html = f'<p>Error fetching news: {news_info.get("error")}</p>'
    else:
        news_html = '<p>No news available</p>'
    
    html = f"""<!DOCTYPE html>
<html lang="en">
    <head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{symbol} - Stock Information</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-zoom@2.0.1/dist/chartjs-plugin-zoom.umd.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-annotation@3.0.1/dist/chartjs-plugin-annotation.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: #f5f5f5;
            color: #333;
            line-height: 1.6;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }}
        .header {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .header h1 {{
            font-size: 32px;
            margin-bottom: 10px;
        }}
        .price-section {{
            display: flex;
            align-items: center;
            gap: 20px;
            margin: 20px 0;
        }}
        .price {{
            font-size: 48px;
            font-weight: bold;
        }}
        .change {{
            font-size: 24px;
            font-weight: 500;
        }}
        .change.positive {{
            color: #16a34a;
        }}
        .change.negative {{
            color: #dc2626;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .metric-card {{
            background: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .metric-label {{
            font-size: 12px;
            color: #666;
            text-transform: uppercase;
            margin-bottom: 5px;
        }}
        .metric-value {{
            font-size: 20px;
            font-weight: 600;
        }}
        .chart-section {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .chart-controls {{
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
        }}
        .chart-btn {{
            padding: 8px 16px;
            border: 1px solid #ddd;
            background: white;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
        }}
        .chart-btn.active {{
            background: #667eea;
            color: white;
            border-color: #667eea;
        }}
        .chart-container {{
            position: relative;
            height: 400px;
        }}
        .data-section {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .data-section h2 {{
            margin-bottom: 15px;
            color: #667eea;
        }}
        .data-table {{
            width: 100%;
            border-collapse: collapse;
        }}
        .data-table th,
        .data-table td {{
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid #eee;
        }}
        .data-table th {{
            background: #f9fafb;
            font-weight: 600;
            color: #666;
        }}
        .status-indicator {{
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            margin-right: 5px;
        }}
        .status-indicator.connected {{
            background: #16a34a;
        }}
        .status-indicator.disconnected {{
            background: #dc2626;
        }}
        .realtime-section {{
            background: #f9fafb;
            padding: 10px;
            border-radius: 4px;
            margin-top: 10px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="header-content-wrapper" style="display: flex; align-items: flex-start; gap: 30px;">
                <div>
                    <h1>{symbol}</h1>
                    <div class="price-section">
                        <div class="price">${current_price_str}</div>
                        <div class="change {change_color}">
                            {change_sign}${abs(price_change):.2f} ({change_sign}{price_change_pct:.2f}%)
                        </div>
                    </div>
                    <div class="realtime-section">
                        <span class="status-indicator disconnected" id="wsStatus"></span>
                        <span id="wsStatusText">Connecting to real-time data...</span>
                        <span id="realtimePrice" style="margin-left: 20px; font-weight: 600;"></span>
                    </div>
                </div>
                <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-label">Market Cap</div>
                <div class="metric-value">{format_value(financial_data.get('market_cap'))}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">P/E Ratio</div>
                <div class="metric-value">{format_value(financial_data.get('price_to_earnings') or financial_data.get('pe_ratio'))}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Day's Range</div>
                <div class="metric-value">{f"{format_value(daily_low)} - {format_value(daily_high)}" if daily_low is not None and daily_high is not None else 'N/A'}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">52 Week High</div>
                <div class="metric-value">{format_value(week_52_high)}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">52 Week Low</div>
                <div class="metric-value">{format_value(week_52_low)}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Volume</div>
                <div class="metric-value">{format_value(volume)}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Implied Volatility</div>
                <div class="metric-value">{format_value((iv_data.get('statistics', {}).get('mean') or iv_data.get('atm_iv', {}).get('mean')) if iv_data else None)}</div>
            </div>
            {f'''
            <div class="metric-card">
                <div class="metric-label">Earnings Date</div>
                <div class="metric-value">{earnings_date_display if earnings_date_display and earnings_date_display != 'N/A' and 'Dividend' not in earnings_date_display and 'Yield' not in earnings_date_display else 'N/A'}</div>
            </div>
            ''' if earnings_date else ''}
                </div>
            </div>
        </div>
        
        <div class="chart-section">
            <h2>Price Chart</h2>
            <div class="chart-controls" style="display: flex; flex-wrap: wrap; gap: 5px; margin-bottom: 10px;">
                <button class="chart-btn active" onclick="switchTimePeriod('1d')" id="btn-1d">1D</button>
                <button class="chart-btn" onclick="switchTimePeriod('1w')" id="btn-1w">1W</button>
                <button class="chart-btn" onclick="switchTimePeriod('1m')" id="btn-1m">1M</button>
                <button class="chart-btn" onclick="switchTimePeriod('3m')" id="btn-3m">3M</button>
                <button class="chart-btn" onclick="switchTimePeriod('6m')" id="btn-6m">6M</button>
                <button class="chart-btn" onclick="switchTimePeriod('ytd')" id="btn-ytd">YTD</button>
                <button class="chart-btn" onclick="switchTimePeriod('1y')" id="btn-1y">1Y</button>
                <button class="chart-btn" onclick="switchTimePeriod('2y')" id="btn-2y">2Y</button>
            </div>
            <div class="chart-container">
                <canvas id="priceChart"></canvas>
                <div id="chartNoDataMessage" style="display: none; text-align: center; padding: 40px; color: #666;">
                    <p>No historical price data available for this symbol.</p>
                    <p style="font-size: 12px; margin-top: 10px;">Try adding <code>?force_fetch=true</code> to the URL to fetch data from the API.</p>
                </div>
            </div>
        </div>
        
        
        {f'''
        <div class="data-section">
            <h2 style="cursor: pointer; user-select: none;" onclick="toggleIVSection()">
                Implied Volatility
                <span id="ivCaret" style="display: inline-block; margin-left: 8px; transition: transform 0.2s;">▶</span>
            </h2>
            <div id="ivContent" style="display: none;">
                <table class="data-table">
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                    </tr>
                    <tr><td>Mean IV</td><td>{format_value(iv_data.get('statistics', {}).get('mean'))}</td></tr>
                    <tr><td>Median IV</td><td>{format_value(iv_data.get('statistics', {}).get('median'))}</td></tr>
                    <tr><td>ATM IV</td><td>{format_value(iv_data.get('atm_iv', {}).get('mean'))}</td></tr>
                    <tr><td>Call IV</td><td>{format_value(iv_data.get('call_iv', {}).get('mean'))}</td></tr>
                    <tr><td>Put IV</td><td>{format_value(iv_data.get('put_iv', {}).get('mean'))}</td></tr>
                    <tr><td>IV Count</td><td>{format_value(iv_data.get('statistics', {}).get('count'))}</td></tr>
                </table>
            </div>
        </div>
        <script>
            function toggleIVSection() {{
                const content = document.getElementById('ivContent');
                const caret = document.getElementById('ivCaret');
                if (content.style.display === 'none') {{
                    content.style.display = 'block';
                    caret.style.transform = 'rotate(90deg)';
                }} else {{
                    content.style.display = 'none';
                    caret.style.transform = 'rotate(0deg)';
                }}
            }}
        </script>
        ''' if iv_data else ''}
        
        {f'''
        <div class="data-section">
            <h2>Options</h2>
            <div id="optionsDisplay">
                {format_options_html(options_data, current_price if isinstance(current_price, (int, float)) else None) if options_data else '<p>No options data available</p>'}
            </div>
        </div>
        ''' if options_data else ''}
        
        {f'''
        <div class="data-section">
            <h2>Latest News</h2>
            <div id="newsDisplay">
                {news_html}
            </div>
        </div>
        ''' if news_info else ''}
    </div>
    
    <script>
        // Global symbol for API requests
        const symbol = "{symbol}";
        
        // Merged chart data - close values & timestamps with source metadata
        const allChartData = {all_chart_data_json};
        const allChartLabels = {all_chart_labels_json};
        const mergedSeries = {merged_series_json};
        
        let currentTimePeriod = '1d';
        
        // Find reference price (previous day's close) for calculating change
        let referencePrice = null;
        if (Array.isArray(mergedSeries) && mergedSeries.length > 0) {{
            // Sort by timestamp to ensure chronological order
            const sortedSeries = [...mergedSeries].sort((a, b) => {{
                const dateA = parseDate(a.timestamp);
                const dateB = parseDate(b.timestamp);
                if (!dateA || !dateB) return 0;
                return dateA.getTime() - dateB.getTime();
            }});
            
            // Get today's date (in local timezone, but we'll compare dates)
            const now = new Date();
            const todayDate = now.toISOString().slice(0, 10); // YYYY-MM-DD
            
            // Find the most recent daily close from a previous day
            let previousDayClose = null;
            let previousDayDate = null;
            
            for (let i = sortedSeries.length - 1; i >= 0; i--) {{
                const r = sortedSeries[i];
                if (!r || !r.timestamp) continue;
                
                const dt = parseDate(r.timestamp);
                if (!dt) continue;
                
                const recordDate = dt.toISOString().slice(0, 10); // YYYY-MM-DD
                
                // If this is from a previous day (before today)
                if (recordDate < todayDate) {{
                    // If it's a daily close or from daily source, use it
                    if (r.is_daily_close || r.source === 'daily') {{
                        previousDayClose = parseFloat(r.close);
                        previousDayDate = recordDate;
                        break;
                    }}
                    // Otherwise, keep track of the latest price from previous day
                    if (!previousDayClose) {{
                        previousDayClose = parseFloat(r.close);
                        previousDayDate = recordDate;
                    }}
                }}
            }}
            
            // If we found a previous day's close, use it
            if (previousDayClose) {{
                referencePrice = previousDayClose;
                console.log('Reference price (previous day close):', referencePrice, 'from date:', previousDayDate);
            }} else {{
                // Fallback: find the most recent daily close regardless of date
                for (let i = sortedSeries.length - 1; i >= 0; i--) {{
                    const r = sortedSeries[i];
                    if (r && (r.is_daily_close || r.source === 'daily')) {{
                        referencePrice = parseFloat(r.close);
                        console.log('Reference price (fallback daily close):', referencePrice);
                        break;
                    }}
                }}
            }}
            
            // Last resort: use first available price
            if (!referencePrice && sortedSeries.length > 0) {{
                referencePrice = parseFloat(sortedSeries[0].close) || null;
                console.log('Reference price (first available):', referencePrice);
            }}
        }}
        
        console.log('Final reference price:', referencePrice);
        
        // Update initial price display with change from reference price if available
        if (referencePrice && referencePrice > 0) {{
            const currentPriceElement = document.querySelector('.price');
            const changeElement = document.querySelector('.change');
            
            if (currentPriceElement && changeElement) {{
                const currentPriceText = currentPriceElement.textContent.replace('$', '').trim();
                const currentPrice = parseFloat(currentPriceText);
                
                if (!isNaN(currentPrice) && currentPrice > 0) {{
                    const change = currentPrice - referencePrice;
                    const changePct = (change / referencePrice) * 100;
                    const changeSign = change >= 0 ? '+' : '';
                    
                    // Update change display
                    changeElement.textContent = `${{changeSign}}$${{Math.abs(change).toFixed(2)}} (${{changeSign}}${{changePct.toFixed(2)}}%)`;
                    changeElement.classList.remove('positive', 'negative');
                    changeElement.classList.add(change >= 0 ? 'positive' : 'negative');
                    
                    console.log('Updated price change display:', {{
                        currentPrice: currentPrice,
                        referencePrice: referencePrice,
                        change: change,
                        changePct: changePct
                    }});
                }}
            }}
        }}
        
        // Debug logging
        console.log('Chart data loaded:', {{
            dataLength: allChartData.length,
            labelsLength: allChartLabels.length,
            sampleData: allChartData.slice(0, 5),
            sampleLabels: allChartLabels.slice(0, 5)
        }});
        
        // Calculate date ranges
        const now = new Date();
        const getDateRange = (period) => {{
            const ranges = {{
                '1d': 1,
                '1w': 7,
                '1m': 30,
                '3m': 90,
                '6m': 180,
                'ytd': Math.floor((now - new Date(now.getFullYear(), 0, 1)) / (1000 * 60 * 60 * 24)),
                '1y': 365,
                '2y': 730
            }};
            return ranges[period] || 365;
        }};
        
        // Helper function to parse date string
        function parseDate(dateStr) {{
            if (!dateStr) return null;
            // Try parsing as-is first
            let date = new Date(dateStr);
            if (!isNaN(date.getTime())) return date;
            
            // Try common date formats
            // Format: YYYY-MM-DD
            if (/^\\d{{4}}-\\d{{2}}-\\d{{2}}$/.test(dateStr)) {{
                date = new Date(dateStr + 'T00:00:00');
                if (!isNaN(date.getTime())) return date;
            }}
            
            // Format: YYYY-MM-DD HH:MM:SS
            if (/^\\d{{4}}-\\d{{2}}-\\d{{2}} \\d{{2}}:\\d{{2}}:\\d{{2}}$/.test(dateStr)) {{
                date = new Date(dateStr.replace(' ', 'T'));
                if (!isNaN(date.getTime())) return date;
            }}
            
            return null;
        }}
        
        // Format label based on period: time-only for 1D, date for longer ranges
        function formatLabel(dateObj, period) {{
            if (!(dateObj instanceof Date) || isNaN(dateObj.getTime())) return '';
            if (period === '1d') {{
                const h = String(dateObj.getHours()).padStart(2, '0');
                const m = String(dateObj.getMinutes()).padStart(2, '0');
                return `${{h}}:${{m}}`;
            }} else {{
                // YYYY-MM-DD
                const y = dateObj.getFullYear();
                const mo = String(dateObj.getMonth() + 1).padStart(2, '0');
                const d = String(dateObj.getDate()).padStart(2, '0');
                return `${{y}}-${{mo}}-${{d}}`;
            }}
        }}

        // Build a downsampled series for a given period using mergedSeries.
        // Rules:
        // - 1D: use intraday data (realtime/hourly/daily) with time-bucketed sampling.
        // - >1D: use one point per day from daily data only, so days are evenly represented.
        function buildSeriesForPeriod(period) {{
            if (!Array.isArray(mergedSeries) || mergedSeries.length === 0) {{
                return {{ labels: [], data: [], dateMarkers: [] }};
            }}

            const days = getDateRange(period);
            const nowMs = Date.now();
            const windowStartMs = nowMs - days * 24 * 60 * 60 * 1000;

            // Multi-day windows: use one point per calendar day (based on the
            // latest available sample for that day, regardless of source).
            if (days > 1) {{
                const dailyMap = new Map(); // key: YYYY-MM-DD -> {{dt, val}}
                for (const r of mergedSeries) {{
                    if (!r || typeof r !== 'object') continue;
                    const dt = parseDate(r.timestamp);
                    if (!dt) continue;
                    const t = dt.getTime();
                    if (t < windowStartMs) continue;
                    const key = dt.toISOString().slice(0, 10); // YYYY-MM-DD
                    const val = Number(r.close);
                    if (Number.isNaN(val)) continue;
                    const existing = dailyMap.get(key);
                    // Keep the latest point for that day
                    if (!existing || dt > existing.dt) {{
                        dailyMap.set(key, {{ dt, val }});
                    }}
                }}

                const entries = Array.from(dailyMap.values()).sort((a, b) => a.dt - b.dt);
                if (entries.length === 0) {{
                    return {{ labels: [], data: [], dateMarkers: [] }};
                }}

                const labels = [];
                const data = [];
                const dateMarkers = [];
                let lastDate = null;
                
                for (let i = 0; i < entries.length; i++) {{
                    const p = entries[i];
                    const currentDate = p.dt.toISOString().slice(0, 10); // YYYY-MM-DD
                    
                    // Check if this is the first occurrence of a new date
                    if (lastDate !== null && currentDate !== lastDate) {{
                        dateMarkers.push(i);
                    }}
                    
                    labels.push(formatLabel(p.dt, period));
                    data.push(p.val);
                    lastDate = currentDate;
                }}
                
                // Only show markers if they won't clutter (less than 50% of labels)
                // But always show at least the first date marker if there are multiple dates
                let finalMarkers = [];
                if (dateMarkers.length <= labels.length * 0.5) {{
                    finalMarkers = dateMarkers;
                }} else if (dateMarkers.length > 0 && labels.length > 10) {{
                    // If too many markers, show only significant ones (first, middle, last)
                    finalMarkers = [
                        dateMarkers[0],
                        ...(dateMarkers.length > 2 ? [dateMarkers[Math.floor(dateMarkers.length / 2)]] : []),
                        dateMarkers[dateMarkers.length - 1]
                    ].filter((v, i, arr) => arr.indexOf(v) === i); // Remove duplicates
                }} else if (dateMarkers.length > 0) {{
                    // For small datasets, show all markers
                    finalMarkers = dateMarkers;
                }}
                
                return {{ labels, data, dateMarkers: finalMarkers }};
            }}

            // 1D window: use full merged intraday data with time-bucketed sampling.
            let windowed = [];
            for (const r of mergedSeries) {{
                if (!r || typeof r !== 'object') continue;
                const dt = parseDate(r.timestamp);
                if (!dt) continue;
                const t = dt.getTime();
                if (t < windowStartMs) continue;
                const val = Number(r.close);
                if (!Number.isNaN(val)) {{
                    windowed.push({{ dt, val }});
                }}
            }}

            if (windowed.length === 0) {{
                return {{ labels: [], data: [], dateMarkers: [] }};
            }}

            // Sort by time
            windowed.sort((a, b) => a.dt - b.dt);

            // Decide maximum number of buckets/points we want to display
            let maxPoints = 600;
            if (windowed.length <= maxPoints) {{
                const labelsDirect = [];
                const dataDirect = [];
                const dateMarkers = [];
                let lastDate = null;
                
                for (let i = 0; i < windowed.length; i++) {{
                    const p = windowed[i];
                    const currentDate = p.dt.toISOString().slice(0, 10); // YYYY-MM-DD
                    
                    // Check if this is the first occurrence of a new date
                    if (lastDate !== null && currentDate !== lastDate) {{
                        dateMarkers.push(i);
                    }}
                    
                    labelsDirect.push(formatLabel(p.dt, period));
                    dataDirect.push(p.val);
                    lastDate = currentDate;
                }}
                
                // Only show markers if they won't clutter (less than 50% of labels)
                const finalMarkers = dateMarkers.length <= labelsDirect.length * 0.5 ? dateMarkers : [];
                
                return {{ labels: labelsDirect, data: dataDirect, dateMarkers: finalMarkers }};
            }}

            const firstMs = windowed[0].dt.getTime();
            const lastMs = windowed[windowed.length - 1].dt.getTime();
            const totalMs = Math.max(lastMs - firstMs, 1);
            const bucketCount = Math.min(maxPoints, windowed.length);
            const bucketSize = totalMs / (bucketCount - 1);

            const buckets = new Array(bucketCount);
            const bucketDates = new Array(bucketCount);
            for (const p of windowed) {{
                const t = p.dt.getTime();
                const rawIndex = (t - firstMs) / bucketSize;
                let idx = Math.round(rawIndex);
                if (idx < 0 || idx >= bucketCount) {{
                    continue;
                }}
                buckets[idx] = p;
                // Store the date for this bucket (use the point's date)
                if (!bucketDates[idx] || p.dt < bucketDates[idx]) {{
                    bucketDates[idx] = p.dt;
                }}
            }}

            const labels = [];
            const data = [];
            const dateMarkers = [];
            let lastDate = null;
            
            for (let i = 0; i < bucketCount; i++) {{
                const point = buckets[i];
                if (!point) continue;
                const bucketCenterMs = firstMs + i * bucketSize;
                const labelDate = bucketDates[i] || new Date(bucketCenterMs);
                const currentDate = labelDate.toISOString().slice(0, 10); // YYYY-MM-DD
                
                // Check if this is the first occurrence of a new date
                if (lastDate !== null && currentDate !== lastDate) {{
                    dateMarkers.push(i);
                }}
                
                labels.push(formatLabel(labelDate, period));
                data.push(point.val);
                lastDate = currentDate;
            }}
            
            // Only show markers if they won't clutter (less than 50% of labels)
            const finalMarkers = dateMarkers.length <= labels.length * 0.5 ? dateMarkers : [];

            return {{ labels, data, dateMarkers: finalMarkers }};
        }}

        const ctx = document.getElementById('priceChart').getContext('2d');
        let priceChart = null;
        
        // Register annotation plugin before creating charts
        function registerAnnotationPlugin() {{
            try {{
                // Chart.js annotation plugin v3 is available as window.Chart.Annotation
                // or as a global variable after loading from CDN
                if (window.Chart && window.Chart.register) {{
                    // Try to find the annotation plugin
                    let annotationPlugin = null;
                    
                    // Check various possible locations
                    if (window.Chart.Annotation) {{
                        annotationPlugin = window.Chart.Annotation;
                    }} else if (window.chartjsPluginAnnotation) {{
                        annotationPlugin = window.chartjsPluginAnnotation;
                    }} else if (window['chartjs-plugin-annotation']) {{
                        annotationPlugin = window['chartjs-plugin-annotation'];
                    }} else if (typeof annotation !== 'undefined') {{
                        annotationPlugin = annotation;
                    }}
                    
                    if (annotationPlugin) {{
                        window.Chart.register(annotationPlugin);
                        console.log('Annotation plugin registered successfully');
                        return true;
                    }} else {{
                        console.warn('Annotation plugin not found. Available globals:', Object.keys(window).filter(k => k.toLowerCase().includes('chart')));
                        return false;
                    }}
                }} else {{
                    console.warn('Chart.js not available or Chart.register not available');
                    return false;
                }}
            }} catch (e) {{
                console.warn('Could not register annotation plugin', e);
                return false;
            }}
        }}
        
        function initChart() {{
            if (priceChart) {{
                priceChart.destroy();
            }}
            
            // Register annotation plugin before creating chart
            registerAnnotationPlugin();
            
            // If no data, show message
            if (!Array.isArray(mergedSeries) || mergedSeries.length === 0) {{
                console.warn('No chart data available');
                const noDataMsg = document.getElementById('chartNoDataMessage');
                if (noDataMsg) {{
                    noDataMsg.style.display = 'block';
                }}
                priceChart = new Chart(ctx, {{
                    type: 'line',
                    data: {{
                        labels: [],
                        datasets: [{{
                            label: '{symbol} Price',
                            data: [],
                            borderColor: '#667eea',
                            backgroundColor: 'rgba(102, 126, 234, 0.1)',
                            borderWidth: 2
                        }}]
                    }},
                    options: {{
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {{
                            legend: {{
                                display: false
                            }},
                            tooltip: {{
                                enabled: false
                            }},
                            zoom: {{
                                zoom: {{
                                    wheel: {{
                                        enabled: true
                                    }},
                                    drag: {{
                                        enabled: true
                                    }},
                                    mode: 'x'
                                }},
                                pan: {{
                                    enabled: true,
                                    mode: 'x'
                                }}
                            }}
                        }},
                        scales: {{
                            y: {{
                                beginAtZero: false
                            }}
                        }}
                    }}
                }});
                return;
            }}
            
            // Hide no data message if we have data
            const noDataMsg = document.getElementById('chartNoDataMessage');
            if (noDataMsg) {{
                noDataMsg.style.display = 'none';
            }}
            
            // Build initial 1D series using merged data (with 30-day rule)
            const initialSeries = buildSeriesForPeriod('1d');
            console.log('Initializing chart with', initialSeries.data.length, 'data points');
            console.log('Date markers found:', initialSeries.dateMarkers);
            
            // Build date marker annotations
            const dateMarkerAnnotations = buildDateMarkerAnnotations(initialSeries.dateMarkers, initialSeries.labels);
            console.log('Date marker annotations:', dateMarkerAnnotations);
            
            // Build plugins object with annotations
            const pluginsConfig = {{
                legend: {{
                    display: false
                }},
                zoom: {{
                    zoom: {{
                        wheel: {{
                            enabled: true
                        }},
                        drag: {{
                            enabled: true
                        }},
                        mode: 'x'
                    }},
                    pan: {{
                        enabled: true,
                        mode: 'x'
                    }}
                }}
            }};
            
            // Add annotations if available
            if (dateMarkerAnnotations.annotations && Object.keys(dateMarkerAnnotations.annotations).length > 0) {{
                // Check if annotation plugin is available
                const hasAnnotationPlugin = window.Chart && 
                    (window.Chart.Annotation || 
                     window.chartjsPluginAnnotation || 
                     window['chartjs-plugin-annotation']);
                
                if (hasAnnotationPlugin) {{
                    pluginsConfig.annotation = {{
                        annotations: dateMarkerAnnotations.annotations
                    }};
                    console.log('Added', Object.keys(dateMarkerAnnotations.annotations).length, 'date marker annotations to chart');
                }} else {{
                    console.warn('Annotation plugin not available - date markers will not be displayed');
                    console.warn('Available Chart.js plugins:', Object.keys(window).filter(k => k.toLowerCase().includes('chart')));
                }}
            }} else {{
                console.log('No date marker annotations to add (dateMarkers:', initialSeries.dateMarkers, ')');
            }}
            
            priceChart = new Chart(ctx, {{
                type: 'line',
                data: {{
                    labels: initialSeries.labels,
                    datasets: [{{
                        label: '{symbol} Price',
                        data: initialSeries.data,
                        borderColor: '#667eea',
                        backgroundColor: 'rgba(102, 126, 234, 0.1)',
                        borderWidth: 2,
                        fill: true,
                        tension: 0.4
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: pluginsConfig,
                    scales: {{
                        y: {{
                            beginAtZero: false
                        }},
                        x: {{
                            ticks: {{
                                maxTicksLimit: 20
                            }}
                        }}
                    }}
                }}
            }});

            // Register zoom plugin for drag-to-zoom selection
            try {{
                const zoomPlugin = window.ChartZoom || window['chartjs-plugin-zoom'] || window.Zoom;
                if (zoomPlugin && window.Chart && typeof window.Chart.register === 'function') {{
                    window.Chart.register(zoomPlugin);
                }}
            }} catch (e) {{
                console.warn('Could not register zoom plugin', e);
            }}
        }}
        
        // Helper function to build annotation configuration from date markers
        function buildDateMarkerAnnotations(dateMarkers, labels) {{
            if (!dateMarkers || dateMarkers.length === 0) {{
                console.log('No date markers to display');
                return {{}};
            }}
            
            console.log('Building annotations for', dateMarkers.length, 'date markers');
            
            const annotations = {{}};
            dateMarkers.forEach((index, idx) => {{
                if (index < 0 || index >= labels.length) return;
                
                // Use unique key for each annotation
                const key = `dateMarker_${{idx}}`;
                const labelValue = labels[index];
                
                // Extract the actual date from mergedSeries for this index
                // The label might be a time (HH:MM) or a date (YYYY-MM-DD)
                let dateLabel = labelValue;
                
                // Check if label is already a date (YYYY-MM-DD format)
                if (/^\\d{{4}}-\\d{{2}}-\\d{{2}}$/.test(labelValue)) {{
                    // Already a date, use it
                    dateLabel = labelValue;
                }} else {{
                    // It's a time, need to get the date from mergedSeries at this index
                    // We'll look up the date by matching the index to the built series
                    if (Array.isArray(mergedSeries) && mergedSeries.length > 0) {{
                        const days = getDateRange(currentTimePeriod);
                        const nowMs = Date.now();
                        const windowStartMs = nowMs - days * 24 * 60 * 60 * 1000;
                        
                        let foundDate = null;
                        
                        if (days > 1) {{
                            // Multi-day: build daily map and get date at index
                            const dailyMap = new Map();
                            for (const r of mergedSeries) {{
                                if (!r || typeof r !== 'object') continue;
                                const dt = parseDate(r.timestamp);
                                if (!dt) continue;
                                const t = dt.getTime();
                                if (t < windowStartMs) continue;
                                const key = dt.toISOString().slice(0, 10);
                                const val = Number(r.close);
                                if (Number.isNaN(val)) continue;
                                const existing = dailyMap.get(key);
                                if (!existing || dt > existing.dt) {{
                                    dailyMap.set(key, {{ dt, val }});
                                }}
                            }}
                            const entries = Array.from(dailyMap.values()).sort((a, b) => a.dt - b.dt);
                            if (index < entries.length) {{
                                foundDate = entries[index].dt;
                            }}
                        }} else {{
                            // 1D: get windowed data and find date at index
                            const windowed = [];
                            for (const r of mergedSeries) {{
                                if (!r || typeof r !== 'object') continue;
                                const dt = parseDate(r.timestamp);
                                if (!dt) continue;
                                const t = dt.getTime();
                                if (t < windowStartMs) continue;
                                const val = Number(r.close);
                                if (!Number.isNaN(val)) {{
                                    windowed.push({{ dt, val }});
                                }}
                            }}
                            windowed.sort((a, b) => a.dt - b.dt);
                            
                            // For 1D, if we have bucketed data, we need to estimate
                            // But for simplicity, use the point at index if available
                            if (index < windowed.length) {{
                                foundDate = windowed[index].dt;
                            }} else if (windowed.length > 0) {{
                                // For bucketed data beyond windowed length, use proportional estimate
                                const maxPoints = 600;
                                if (windowed.length > maxPoints) {{
                                    const firstMs = windowed[0].dt.getTime();
                                    const lastMs = windowed[windowed.length - 1].dt.getTime();
                                    const totalMs = Math.max(lastMs - firstMs, 1);
                                    const bucketCount = Math.min(maxPoints, windowed.length);
                                    const bucketSize = totalMs / (bucketCount - 1);
                                    const bucketCenterMs = firstMs + index * bucketSize;
                                    // Find closest point
                                    let closestPoint = windowed[0];
                                    let minDiff = Math.abs(windowed[0].dt.getTime() - bucketCenterMs);
                                    for (const p of windowed) {{
                                        const diff = Math.abs(p.dt.getTime() - bucketCenterMs);
                                        if (diff < minDiff) {{
                                            minDiff = diff;
                                            closestPoint = p;
                                        }}
                                    }}
                                    foundDate = closestPoint.dt;
                                }} else {{
                                    // Use last point if index is beyond
                                    foundDate = windowed[windowed.length - 1].dt;
                                }}
                            }}
                        }}
                        
                        // Format the date for display
                        if (foundDate) {{
                            const dateStr = foundDate.toISOString().slice(0, 10); // YYYY-MM-DD
                            dateLabel = dateStr;
                        }}
                    }}
                }}
                
                // For category scale, use xMin/xMax with index
                annotations[key] = {{
                    type: 'line',
                    xMin: index,
                    xMax: index,
                    borderColor: 'rgba(100, 100, 100, 0.7)',
                    borderWidth: 2,
                    borderDash: [8, 4],
                    label: {{
                        display: true,
                        content: dateLabel,
                        position: 'start',
                        yAdjust: 10,
                        backgroundColor: 'rgba(100, 100, 100, 0.9)',
                        color: '#fff',
                        font: {{
                            size: 11,
                            weight: 'bold'
                        }},
                        padding: {{
                            top: 5,
                            bottom: 5,
                            left: 8,
                            right: 8
                        }},
                        textAlign: 'center'
                    }}
                }};
            }});
            
            console.log('Created', Object.keys(annotations).length, 'annotations');
            return {{
                annotations: annotations
            }};
        }}
        
        function switchTimePeriod(period) {{
            currentTimePeriod = period;
            document.querySelectorAll('[id^="btn-"]').forEach(btn => {{
                if (btn.id.startsWith('btn-1d') || btn.id.startsWith('btn-1w') || btn.id.startsWith('btn-1m') || 
                    btn.id.startsWith('btn-3m') || btn.id.startsWith('btn-6m') || btn.id.startsWith('btn-ytd') || 
                    btn.id.startsWith('btn-1y') || btn.id.startsWith('btn-2y')) {{
                    btn.classList.remove('active');
                }}
            }});
            document.getElementById(`btn-${{period}}`).classList.add('active');
            
            const series = buildSeriesForPeriod(period);
            if (!priceChart || !series.data || series.data.length === 0) {{
                console.warn('Cannot switch time period: chart not initialized or no data for this period');
                return;
            }}

            // Reset zoom on period change so selection always reflects the new window
            if (typeof priceChart.resetZoom === 'function') {{
                priceChart.resetZoom();
            }}

            // Update chart data - remove any extra datasets (like regression lines) and keep only the main price dataset
            priceChart.data.labels = series.labels;
            // Keep only the first dataset (main price line), remove any others
            if (priceChart.data.datasets.length > 1) {{
                priceChart.data.datasets = [priceChart.data.datasets[0]];
            }}
            priceChart.data.datasets[0].data = series.data;
            
            // Update date marker annotations
            const dateMarkerAnnotations = buildDateMarkerAnnotations(series.dateMarkers, series.labels);
            if (priceChart.options.plugins) {{
                // Remove old date marker annotations
                if (priceChart.options.plugins.annotation && priceChart.options.plugins.annotation.annotations) {{
                    Object.keys(priceChart.options.plugins.annotation.annotations).forEach(key => {{
                        if (key.startsWith('dateMarker_')) {{
                            delete priceChart.options.plugins.annotation.annotations[key];
                        }}
                    }});
                }}
                // Add new annotations
                if (dateMarkerAnnotations.annotations && Object.keys(dateMarkerAnnotations.annotations).length > 0) {{
                    if (!priceChart.options.plugins.annotation) {{
                        priceChart.options.plugins.annotation = {{}};
                    }}
                    if (!priceChart.options.plugins.annotation.annotations) {{
                        priceChart.options.plugins.annotation.annotations = {{}};
                    }}
                    Object.assign(priceChart.options.plugins.annotation.annotations, dateMarkerAnnotations.annotations);
                    console.log('Updated date marker annotations:', Object.keys(dateMarkerAnnotations.annotations).length);
                }} else {{
                    // Clear annotations if none to show
                    if (priceChart.options.plugins.annotation && priceChart.options.plugins.annotation.annotations) {{
                        Object.keys(priceChart.options.plugins.annotation.annotations).forEach(key => {{
                            if (key.startsWith('dateMarker_')) {{
                                delete priceChart.options.plugins.annotation.annotations[key];
                            }}
                        }});
                    }}
                }}
            }}
            
            priceChart.update();
        }}
        
        // Initialize chart after page loads
        if (document.readyState === 'loading') {{
            document.addEventListener('DOMContentLoaded', initChart);
        }} else {{
            // DOM already loaded
            initChart();
        }}
        
        // WebSocket connection for real-time updates
        // Use the same port as the current page URL (proxy will route to backend)
        const wsPort = window.location.port || (window.location.protocol === 'https:' ? '443' : '80');
        let ws = null;
        let reconnectAttempts = 0;
        const maxReconnectAttempts = 5;
        
        function connectWebSocket() {{
            try {{
                // Connect to WebSocket on same host:port as page (proxy routes to backend:9102)
                const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                const host = window.location.hostname || 'localhost';
                const wsUrl = `${{protocol}}//${{host}}:${{wsPort}}/stock_info/ws?symbol={symbol}`;
                console.log('Connecting to WebSocket:', wsUrl);
                ws = new WebSocket(wsUrl);
                
                ws.onopen = function() {{
                    console.log('WebSocket connected');
                    document.getElementById('wsStatus').classList.remove('disconnected');
                    document.getElementById('wsStatus').classList.add('connected');
                    document.getElementById('wsStatusText').textContent = 'Connected to real-time data';
                    reconnectAttempts = 0;
                }};
                
                ws.onmessage = function(event) {{
                    try {{
                        const data = JSON.parse(event.data);
                        if (data.symbol === '{symbol}' && data.data) {{
                            updateRealtimePrice(data.data);
                        }}
                    }} catch (e) {{
                        console.error('Error parsing WebSocket message:', e);
                    }}
                }};
                
                ws.onerror = function(error) {{
                    console.error('WebSocket error:', error);
                    document.getElementById('wsStatus').classList.remove('connected');
                    document.getElementById('wsStatus').classList.add('disconnected');
                    document.getElementById('wsStatusText').textContent = 'Connection error';
                }};
                
                ws.onclose = function() {{
                    console.log('WebSocket closed');
                    document.getElementById('wsStatus').classList.remove('connected');
                    document.getElementById('wsStatus').classList.add('disconnected');
                    document.getElementById('wsStatusText').textContent = 'Disconnected';
                    
                    // Attempt to reconnect
                    if (reconnectAttempts < maxReconnectAttempts) {{
                        reconnectAttempts++;
                        setTimeout(connectWebSocket, 3000);
                    }}
                }};
            }} catch (e) {{
                console.error('Error connecting WebSocket:', e);
                document.getElementById('wsStatusText').textContent = 'WebSocket not available';
            }}
        }}
        
        function updateRealtimePrice(data) {{
            if (data.type === 'quote' && data.payload && data.payload.length > 0) {{
                const quote = data.payload[0];
                const price = quote.bid_price || quote.price;
                if (price && priceChart) {{
                    const priceFloat = parseFloat(price);
                    if (isNaN(priceFloat)) return;
                    
                    // Update the small realtime price display
                    document.getElementById('realtimePrice').textContent = `Real-time: ${{priceFloat.toFixed(2)}}`;
                    
                    // Update the main price display
                    const priceElement = document.querySelector('.price');
                    if (priceElement) {{
                        priceElement.textContent = `$${{priceFloat.toFixed(2)}}`;
                    }}
                    
                    // Calculate change and change percentage from reference price
                    let change = 0;
                    let changePct = 0;
                    if (referencePrice && !isNaN(referencePrice) && referencePrice > 0) {{
                        change = priceFloat - referencePrice;
                        changePct = (change / referencePrice) * 100;
                    }}
                    
                    // Update the change display
                    const changeElement = document.querySelector('.change');
                    if (changeElement) {{
                        const changeSign = change >= 0 ? '+' : '';
                        changeElement.textContent = `${{changeSign}}$${{Math.abs(change).toFixed(2)}} (${{changeSign}}${{changePct.toFixed(2)}}%)`;
                        // Update color class
                        changeElement.classList.remove('positive', 'negative');
                        changeElement.classList.add(change >= 0 ? 'positive' : 'negative');
                    }}
                    
                    // Add new realtime point to mergedSeries array
                    const newPoint = {{
                        timestamp: quote.timestamp,
                        close: priceFloat,
                        source: 'realtime',
                        is_daily_open: false,
                        is_daily_close: false
                    }};
                    
                    // Add to mergedSeries (append, then re-sort by timestamp if needed)
                    mergedSeries.push(newPoint);
                    
                    // Rebuild the chart data using the downsampling logic
                    const sampled = buildSeriesForPeriod(currentTimePeriod);
                    
                    // Update chart with newly sampled data - remove any extra datasets
                    priceChart.data.labels = sampled.labels;
                    // Keep only the first dataset (main price line), remove any others
                    if (priceChart.data.datasets.length > 1) {{
                        priceChart.data.datasets = [priceChart.data.datasets[0]];
                    }}
                    priceChart.data.datasets[0].data = sampled.data;
                    
                    // Update date marker annotations
                    const dateMarkerAnnotations = buildDateMarkerAnnotations(sampled.dateMarkers, sampled.labels);
                    if (priceChart.options.plugins) {{
                        // Remove old date marker annotations
                        if (priceChart.options.plugins.annotation && priceChart.options.plugins.annotation.annotations) {{
                            Object.keys(priceChart.options.plugins.annotation.annotations).forEach(key => {{
                                if (key.startsWith('dateMarker_')) {{
                                    delete priceChart.options.plugins.annotation.annotations[key];
                                }}
                            }});
                        }}
                        // Add new annotations
                        if (dateMarkerAnnotations.annotations && Object.keys(dateMarkerAnnotations.annotations).length > 0) {{
                            if (!priceChart.options.plugins.annotation) {{
                                priceChart.options.plugins.annotation = {{}};
                            }}
                            if (!priceChart.options.plugins.annotation.annotations) {{
                                priceChart.options.plugins.annotation.annotations = {{}};
                            }}
                            Object.assign(priceChart.options.plugins.annotation.annotations, dateMarkerAnnotations.annotations);
                        }} else {{
                            // Clear annotations if none to show
                            if (priceChart.options.plugins.annotation && priceChart.options.plugins.annotation.annotations) {{
                                Object.keys(priceChart.options.plugins.annotation.annotations).forEach(key => {{
                                    if (key.startsWith('dateMarker_')) {{
                                        delete priceChart.options.plugins.annotation.annotations[key];
                                    }}
                                }});
                            }}
                        }}
                    }}
                    
                    priceChart.update('none');
                }}
            }}
        }}
        
        // Connect on page load
        connectWebSocket();
        
        // Cleanup on page unload
        window.addEventListener('beforeunload', function() {{
            if (ws) {{
                ws.close();
            }}
        }});
    </script>
</body>
</html>"""
    
    return html


