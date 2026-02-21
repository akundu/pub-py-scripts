"""Display and formatting utilities for stock data."""
from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def normalize_timezone_string(tz_string: str) -> str:
    """Convert common timezone abbreviations to proper pytz timezone strings.
    
    Args:
        tz_string: Timezone string or abbreviation
        
    Returns:
        Normalized timezone string
    """
    # Common timezone abbreviation mappings
    tz_abbreviations = {
        # US Timezones
        'EST': 'America/New_York',
        'EDT': 'America/New_York', 
        'CST': 'America/Chicago',
        'CDT': 'America/Chicago',
        'MST': 'America/Denver',
        'MDT': 'America/Denver',
        'PST': 'America/Los_Angeles',
        'PDT': 'America/Los_Angeles',
        'AKST': 'America/Anchorage',
        'AKDT': 'America/Anchorage',
        'HST': 'Pacific/Honolulu',
        'HAST': 'Pacific/Honolulu',
        
        # Other common abbreviations
        'UTC': 'UTC',
        'GMT': 'Europe/London',
        'BST': 'Europe/London',
        'CET': 'Europe/Paris',
        'CEST': 'Europe/Paris',
        'JST': 'Asia/Tokyo',
        'CST_CN': 'Asia/Shanghai',  # China Standard Time
        'IST': 'Asia/Kolkata',      # India Standard Time
        'AEST': 'Australia/Sydney',
        'AEDT': 'Australia/Sydney',
    }
    
    # Check if it's already a proper timezone string (contains '/')
    if '/' in tz_string:
        return tz_string
    
    # Convert abbreviation to proper timezone
    normalized = tz_abbreviations.get(tz_string.upper())
    if normalized:
        return normalized
    
    # If not found, return as-is (might be a valid pytz string)
    return tz_string


def normalize_index_timestamp(idx_value) -> datetime | None:
    """Normalize various index timestamp formats into a timezone-aware UTC datetime.
    
    Args:
        idx_value: Timestamp in various formats (pandas Timestamp, numpy datetime64, str, int)
        
    Returns:
        Timezone-aware UTC datetime or None if conversion fails
    """
    from datetime import timezone
    
    if idx_value is None:
        return None
    try:
        if isinstance(idx_value, datetime):
            dt = idx_value
        elif isinstance(idx_value, (int, float)):
            # Check if it's a date in YYYYMMDD format (8 digits)
            int_val = int(idx_value)
            if 19000101 <= int_val <= 99991231:
                # Likely a date in YYYYMMDD format
                date_str = str(int_val)
                if len(date_str) == 8:
                    dt = datetime.strptime(date_str, '%Y%m%d')
                else:
                    # Try pandas conversion for other integer formats
                    dt = pd.to_datetime(idx_value).to_pydatetime()
            else:
                # Not a date format - could be nanoseconds, milliseconds, or seconds since epoch
                # Try pandas conversion, but validate the result
                dt = pd.to_datetime(idx_value).to_pydatetime()
                # If the result is before 1900, it's likely wrong (probably epoch time misinterpretation)
                if dt.year < 1900:
                    logging.warning(f"Timestamp normalization: integer {idx_value} converted to {dt} (before 1900, likely incorrect). Returning None.")
                    return None
        else:
            # Try pandas conversion for strings and other types
            dt = pd.to_datetime(idx_value).to_pydatetime()
            # Validate the result
            if dt.year < 1900:
                logging.warning(f"Timestamp normalization: {idx_value} (type {type(idx_value)}) converted to {dt} (before 1900, likely incorrect). Returning None.")
                return None
        
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        return dt
    except Exception as e:
        logging.debug(f"Error normalizing timestamp {idx_value} (type {type(idx_value)}): {e}")
        return None


def format_price_block(price_info: dict, target_tz: str | None = 'America/New_York') -> list[str]:
    """Format price information into display lines.
    
    Args:
        price_info: Dictionary with price, bid_price, ask_price, timestamp
        target_tz: Target timezone for timestamp display
        
    Returns:
        List of formatted strings
    """
    from datetime import timezone
    import pytz
    
    lines: list[str] = []
    try:
        price = price_info.get('price')
        bid = price_info.get('bid_price')
        ask = price_info.get('ask_price')
        ts = price_info.get('timestamp')
        
        # Handle timestamp - check if it's a valid datetime string or 'N/A'
        dt = None
        if ts is not None and ts != 'N/A':
            if isinstance(ts, str):
                try:
                    # Try to parse ISO format string
                    dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
                except (ValueError, AttributeError):
                    # If parsing fails, try to parse as datetime object
                    try:
                        dt = pd.to_datetime(ts).to_pydatetime()
                    except (ValueError, TypeError):
                        dt = None
            elif isinstance(ts, datetime):
                dt = ts
            else:
                dt = _normalize_index_timestamp(ts)
        
        # If we couldn't parse the timestamp, use current time or 'N/A'
        if dt is None:
            ts_str = "N/A"
        else:
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            if target_tz:
                tzname = normalize_timezone_string(target_tz)
                dt_disp = dt.astimezone(pytz.timezone(tzname))
            else:
                dt_disp = dt
            ts_str = dt_disp.strftime('%Y-%m-%d %H:%M:%S %Z')
        
        def fmt(x):
            try:
                return f"{float(x):.2f}" if x is not None else "-"
            except Exception:
                return "-"
        lines.append(f"Price: {fmt(price)}  Bid: {fmt(bid)}  Ask: {fmt(ask)}  Time: {ts_str}")
    except Exception as e:
        lines.append(f"Realtime formatting error: {e}")
    return lines


def format_market_cap(value: float) -> str:
    """Format market cap value with appropriate suffix (T/B/M).
    
    Args:
        value: Market cap value
        
    Returns:
        Formatted string
    """
    if value >= 1e12:
        return f"${value/1e12:.2f}T"
    elif value >= 1e9:
        return f"${value/1e9:.2f}B"
    elif value >= 1e6:
        return f"${value/1e6:.2f}M"
    else:
        return f"${value:,.0f}"


def format_shares(value: float) -> str:
    """Format shares outstanding with appropriate suffix (B/M).
    
    Args:
        value: Shares outstanding
        
    Returns:
        Formatted string
    """
    if value >= 1e9:
        return f"{value/1e9:.2f}B"
    elif value >= 1e6:
        return f"{value/1e6:.2f}M"
    else:
        return f"{value:,.0f}"


def display_valuation_ratios(latest_entry: pd.Series) -> None:
    """Display valuation ratios section.
    
    Args:
        latest_entry: Latest financial data entry
    """
    print(f"\n{'Valuation Ratios:':<30}")
    print(f"{'─'*80}")
    if 'price_to_earnings' in latest_entry and pd.notna(latest_entry['price_to_earnings']):
        print(f"  {'P/E Ratio:':<30} {latest_entry['price_to_earnings']}")
    if 'price_to_book' in latest_entry and pd.notna(latest_entry['price_to_book']):
        print(f"  {'P/B Ratio:':<30} {latest_entry['price_to_book']}")
    if 'price_to_sales' in latest_entry and pd.notna(latest_entry['price_to_sales']):
        print(f"  {'P/S Ratio:':<30} {latest_entry['price_to_sales']}")
    if 'peg_ratio' in latest_entry and pd.notna(latest_entry['peg_ratio']):
        print(f"  {'PEG Ratio:':<30} {latest_entry['peg_ratio']}")
    if 'price_to_cash_flow' in latest_entry and pd.notna(latest_entry['price_to_cash_flow']):
        print(f"  {'P/CF Ratio:':<30} {latest_entry['price_to_cash_flow']}")
    if 'price_to_free_cash_flow' in latest_entry and pd.notna(latest_entry['price_to_free_cash_flow']):
        print(f"  {'P/FCF Ratio:':<30} {latest_entry['price_to_free_cash_flow']}")
    if 'ev_to_sales' in latest_entry and pd.notna(latest_entry['ev_to_sales']):
        print(f"  {'EV/Sales:':<30} {latest_entry['ev_to_sales']}")
    if 'ev_to_ebitda' in latest_entry and pd.notna(latest_entry['ev_to_ebitda']):
        print(f"  {'EV/EBITDA:':<30} {latest_entry['ev_to_ebitda']}")


def display_profitability_ratios(latest_entry: pd.Series) -> None:
    """Display profitability ratios section.
    
    Args:
        latest_entry: Latest financial data entry
    """
    print(f"\n{'Profitability Ratios:':<30}")
    print(f"{'─'*80}")
    if 'return_on_equity' in latest_entry and pd.notna(latest_entry['return_on_equity']):
        print(f"  {'ROE:':<30} {latest_entry['return_on_equity']}")
    if 'return_on_assets' in latest_entry and pd.notna(latest_entry['return_on_assets']):
        print(f"  {'ROA:':<30} {latest_entry['return_on_assets']}")
    if 'profit_margin' in latest_entry and pd.notna(latest_entry['profit_margin']):
        print(f"  {'Profit Margin:':<30} {latest_entry['profit_margin']}")
    if 'gross_margin' in latest_entry and pd.notna(latest_entry['gross_margin']):
        print(f"  {'Gross Margin:':<30} {latest_entry['gross_margin']}")
    if 'operating_margin' in latest_entry and pd.notna(latest_entry['operating_margin']):
        print(f"  {'Operating Margin:':<30} {latest_entry['operating_margin']}")


def display_liquidity_ratios(latest_entry: pd.Series) -> None:
    """Display liquidity ratios section.
    
    Args:
        latest_entry: Latest financial data entry
    """
    print(f"\n{'Liquidity Ratios:':<30}")
    print(f"{'─'*80}")
    # Handle both display names (current, quick, cash) and DB names (current_ratio, quick_ratio, cash_ratio)
    current_ratio = latest_entry.get('current') if 'current' in latest_entry else latest_entry.get('current_ratio')
    if current_ratio is not None and pd.notna(current_ratio):
        print(f"  {'Current Ratio:':<30} {current_ratio}")
    quick_ratio = latest_entry.get('quick') if 'quick' in latest_entry else latest_entry.get('quick_ratio')
    if quick_ratio is not None and pd.notna(quick_ratio):
        print(f"  {'Quick Ratio:':<30} {quick_ratio}")
    cash_ratio = latest_entry.get('cash') if 'cash' in latest_entry else latest_entry.get('cash_ratio')
    if cash_ratio is not None and pd.notna(cash_ratio):
        print(f"  {'Cash Ratio:':<30} {cash_ratio}")


def display_leverage_ratios(latest_entry: pd.Series) -> None:
    """Display leverage ratios section.
    
    Args:
        latest_entry: Latest financial data entry
    """
    print(f"\n{'Leverage Ratios:':<30}")
    print(f"{'─'*80}")
    if 'debt_to_equity' in latest_entry and pd.notna(latest_entry['debt_to_equity']):
        print(f"  {'Debt-to-Equity:':<30} {latest_entry['debt_to_equity']}")
    if 'debt_to_assets' in latest_entry and pd.notna(latest_entry['debt_to_assets']):
        print(f"  {'Debt-to-Assets:':<30} {latest_entry['debt_to_assets']}")


def display_market_data(latest_entry: pd.Series) -> None:
    """Display market data section.
    
    Args:
        latest_entry: Latest financial data entry
    """
    print(f"\n{'Market Data:':<30}")
    print(f"{'─'*80}")
    if 'market_cap' in latest_entry and pd.notna(latest_entry['market_cap']):
        market_cap = float(latest_entry['market_cap'])
        print(f"  {'Market Cap:':<30} {format_market_cap(market_cap)}")
    if 'enterprise_value' in latest_entry and pd.notna(latest_entry['enterprise_value']):
        ev = float(latest_entry['enterprise_value'])
        print(f"  {'Enterprise Value:':<30} {format_market_cap(ev)}")
    if 'shares_outstanding' in latest_entry and pd.notna(latest_entry['shares_outstanding']):
        shares = float(latest_entry['shares_outstanding'])
        print(f"  {'Shares Outstanding:':<30} {format_shares(shares)}")
    if 'dividend_yield' in latest_entry and pd.notna(latest_entry['dividend_yield']):
        yield_val = float(latest_entry['dividend_yield']) * 100
        print(f"  {'Dividend Yield:':<30} {yield_val:.2f}%")


def display_cash_flow(latest_entry: pd.Series) -> None:
    """Display cash flow section.
    
    Args:
        latest_entry: Latest financial data entry
    """
    print(f"\n{'Cash Flow:':<30}")
    print(f"{'─'*80}")
    if 'free_cash_flow' in latest_entry and pd.notna(latest_entry['free_cash_flow']):
        fcf = float(latest_entry['free_cash_flow'])
        print(f"  {'Free Cash Flow:':<30} {format_market_cap(fcf)}")
    if 'operating_cash_flow' in latest_entry and pd.notna(latest_entry['operating_cash_flow']):
        ocf = float(latest_entry['operating_cash_flow'])
        print(f"  {'Operating Cash Flow:':<30} {format_market_cap(ocf)}")


def display_iv_analysis(latest_entry: pd.Series, log_level: str = "INFO") -> None:
    """Display IV analysis section.
    
    Args:
        latest_entry: Latest financial data entry
        log_level: Logging level
    """
    import logging
    logger = logging.getLogger(__name__)
    
    print(f"\n{'IV Analysis:':<30}")
    print(f"{'─'*80}")
    
    # Debug: Log what columns we have
    if log_level == "DEBUG":
        logger.debug(f"[DISPLAY_IV] Columns in latest_entry: {list(latest_entry.index) if hasattr(latest_entry, 'index') else list(latest_entry.keys())}")
        logger.debug(f"[DISPLAY_IV] Has iv_analysis_json: {'iv_analysis_json' in latest_entry}")
        if 'iv_analysis_json' in latest_entry:
            json_val = latest_entry.get('iv_analysis_json')
            logger.debug(f"[DISPLAY_IV] iv_analysis_json value type: {type(json_val)}")
            logger.debug(f"[DISPLAY_IV] iv_analysis_json is not None: {json_val is not None}")
            logger.debug(f"[DISPLAY_IV] iv_analysis_json is not empty string: {json_val != '' if isinstance(json_val, str) else 'N/A'}")
            if json_val:
                logger.debug(f"[DISPLAY_IV] iv_analysis_json length: {len(str(json_val))} chars")
                logger.debug(f"[DISPLAY_IV] iv_analysis_json first 200 chars: {str(json_val)[:200]}")
    
    # Check if IV analysis is in the iv_analysis_json column
    # Handle both Series (with .get()) and dict access
    iv_json_val = None
    if hasattr(latest_entry, 'get'):
        iv_json_val = latest_entry.get('iv_analysis_json')
    elif 'iv_analysis_json' in latest_entry:
        iv_json_val = latest_entry['iv_analysis_json']
    elif hasattr(latest_entry, '__getitem__'):
        try:
            iv_json_val = latest_entry['iv_analysis_json']
        except (KeyError, IndexError):
            pass
    
    # Also check if it's in the index (for Series)
    if iv_json_val is None and hasattr(latest_entry, 'index'):
        if 'iv_analysis_json' in latest_entry.index:
            iv_json_val = latest_entry.loc['iv_analysis_json'] if hasattr(latest_entry, 'loc') else latest_entry['iv_analysis_json']
    
    if iv_json_val and (isinstance(iv_json_val, str) and len(iv_json_val.strip()) > 0):
        try:
            # Handle case where it might already be a dict
            if isinstance(iv_json_val, dict):
                iv_analysis = iv_json_val
            else:
                iv_analysis = json.loads(iv_json_val)
            metrics = iv_analysis.get('metrics', {})
            strategy = iv_analysis.get('strategy', {})
            
            # Display all metrics
            if metrics.get('iv_30d'):
                print(f"  {'30-day IV:':<30} {metrics.get('iv_30d', 'N/A')}")
            if metrics.get('iv_90d'):
                print(f"  {'90-day IV:':<30} {metrics.get('iv_90d', 'N/A')}")
            if metrics.get('hv_1yr_range'):
                print(f"  {'1-Year HV Range:':<30} {metrics.get('hv_1yr_range', 'N/A')}")
            if metrics.get('rank') is not None:
                print(f"  {'IV Rank (30-day):':<30} {metrics.get('rank', 'N/A')}")
            if metrics.get('rank_90d') is not None:
                print(f"  {'IV Rank (90-day):':<30} {metrics.get('rank_90d', 'N/A')}")
            if metrics.get('rank_diff') is not None:
                rank_diff = metrics.get('rank_diff')
                diff_label = 'Rank Ratio (30d/90d):'
                print(f"  {diff_label:<30} {rank_diff:.3f}")
            if metrics.get('roll_yield'):
                print(f"  {'Roll Yield:':<30} {metrics.get('roll_yield', 'N/A')}")
            if metrics.get('realized_vol_30d'):
                print(f"  {'30-day Realized Vol:':<30} {metrics.get('realized_vol_30d', 'N/A')}")
            if metrics.get('realized_vol_90d'):
                print(f"  {'90-day Realized Vol:':<30} {metrics.get('realized_vol_90d', 'N/A')}")
            if metrics.get('iv_percentile_30d'):
                print(f"  {'30-day IV Percentile:':<30} {metrics.get('iv_percentile_30d', 'N/A')}")
            if metrics.get('iv_percentile_90d'):
                print(f"  {'90-day IV Percentile:':<30} {metrics.get('iv_percentile_90d', 'N/A')}")
            
            # Display relative_rank (can be at top level or in metrics)
            relative_rank = iv_analysis.get('relative_rank') or metrics.get('relative_rank')
            if relative_rank is not None:
                print(f"  {'Relative Rank (vs VOO):':<30} {relative_rank}")
            
            # Display strategy information
            if strategy.get('recommendation'):
                print(f"  {'Strategy:':<30} {strategy.get('recommendation', 'N/A')}")
            if strategy.get('risk_score') is not None:
                print(f"  {'Risk Score:':<30} {strategy.get('risk_score', 'N/A')}")
            if strategy.get('confidence'):
                print(f"  {'Confidence:':<30} {strategy.get('confidence', 'N/A')}")
            if strategy.get('notes'):
                notes = strategy.get('notes', '')
                if notes:
                    # Handle multi-line notes - ensure notes is a string
                    if isinstance(notes, dict):
                        # If notes is a dict, convert to string representation
                        notes = str(notes)
                    elif not isinstance(notes, str):
                        # Convert other types to string
                        notes = str(notes)
                    
                    # Handle multi-line notes
                    note_lines = notes.split('\n')
                    print(f"  {'Notes:':<30} {note_lines[0] if note_lines else 'N/A'}")
                    for note_line in note_lines[1:]:
                        if note_line.strip():
                            print(f"  {'':<30} {note_line}")
        except (json.JSONDecodeError, TypeError) as e:
            if log_level == "DEBUG":
                logger.debug(f"Could not parse IV analysis JSON: {e}")
                logger.debug(f"Raw iv_analysis_json value: {repr(iv_json_val)}")
                import traceback
                logger.debug(f"Traceback: {traceback.format_exc()}")
            print(f"  {'IV Analysis:':<30} Error parsing IV analysis data: {e}")
    # Also check if IV data is in separate columns (for backwards compatibility)
    elif 'iv_30d' in latest_entry and pd.notna(latest_entry['iv_30d']):
        print(f"  {'30-day IV:':<30} {latest_entry['iv_30d']}")
    elif 'iv_90d' in latest_entry and pd.notna(latest_entry['iv_90d']):
        print(f"  {'90-day IV:':<30} {latest_entry['iv_90d']}")
    else:
        # Debug: Log why IV analysis wasn't found (always log, not just in DEBUG mode)
        has_iv_json_col = 'iv_analysis_json' in latest_entry
        if hasattr(latest_entry, 'index'):
            has_iv_json_col = has_iv_json_col or 'iv_analysis_json' in latest_entry.index
        iv_json_val_check = None
        if hasattr(latest_entry, 'get'):
            iv_json_val_check = latest_entry.get('iv_analysis_json')
        elif 'iv_analysis_json' in latest_entry:
            iv_json_val_check = latest_entry['iv_analysis_json']
        elif hasattr(latest_entry, 'index') and 'iv_analysis_json' in latest_entry.index:
            iv_json_val_check = latest_entry.loc['iv_analysis_json'] if hasattr(latest_entry, 'loc') else latest_entry['iv_analysis_json']
        
        # Always log this as a warning to help diagnose
        logger.warning(f"[DISPLAY_IV] IV analysis not found for latest_entry - has_iv_json_col: {has_iv_json_col}, iv_json_val: {repr(iv_json_val_check) if iv_json_val_check is not None else 'None'}, type: {type(iv_json_val_check)}")
        if hasattr(latest_entry, 'index'):
            available_cols = list(latest_entry.index)
            iv_cols = [c for c in available_cols if 'iv' in str(c).lower()]
            logger.warning(f"[DISPLAY_IV] Available columns ({len(available_cols)}): {available_cols[:20]}...")
            logger.warning(f"[DISPLAY_IV] IV-related columns: {iv_cols}")
        else:
            logger.warning(f"[DISPLAY_IV] Available keys: {list(latest_entry.keys())[:20]}...")
        print(f"  {'IV Analysis:':<30} Not available (use --fetch-iv with --fetch-ratios to calculate)")


def display_historical_data_info(financial_data: pd.DataFrame, log_level: str = "INFO") -> None:
    """Display historical data information section.
    
    Args:
        financial_data: Financial data DataFrame
        log_level: Logging level
    """
    if len(financial_data) > 1:
        print(f"\n{'Historical Data:':<30}")
        print(f"{'─'*80}")
        print(f"  {'Total Records:':<30} {len(financial_data)}")
        if 'date' in financial_data.columns:
            try:
                dates = financial_data['date'].dropna()
                if not dates.empty:
                    # Convert to string/datetime if needed, handling various types
                    date_values = []
                    for date_val in dates:
                        if isinstance(date_val, (str, pd.Timestamp, datetime)):
                            date_values.append(date_val)
                        elif isinstance(date_val, dict):
                            # Skip dict values
                            continue
                        else:
                            try:
                                date_values.append(str(date_val))
                            except Exception:
                                continue
                    
                    if date_values:
                        # Convert to pandas Series for min/max operations
                        date_series = pd.Series(date_values)
                        # Try to convert to datetime if they're strings
                        try:
                            date_series = pd.to_datetime(date_series)
                        except Exception:
                            pass
                        min_date = date_series.min()
                        max_date = date_series.max()
                        print(f"  {'Date Range:':<30} {min_date} to {max_date}")
            except Exception as date_error:
                logger.debug(f"Could not process date range: {date_error}")
                # Silently skip date range display if there's an error

