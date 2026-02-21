"""
Capital lifecycle management utilities for credit spread analysis.

Handles position capital tracking, filtering, and lifecycle management.
"""

from datetime import datetime, date, timedelta, timezone
from typing import Dict, List, Optional, Tuple, Any
import logging
import pandas as pd

from .timezone_utils import (
    normalize_timestamp,
    get_eod_time,
    get_calendar_date,
    get_timezone,
)


def calculate_position_capital(result: Dict, output_tz=None) -> Tuple[float, date]:
    """Calculate position capital (max loss exposure) and get calendar date.
    
    Args:
        result: Result dictionary from analyze_interval
        output_tz: Output timezone for date calculation
    
    Returns:
        Tuple of (position_capital, calendar_date)
    """
    best_spread = result['best_spread']
    num_contracts = best_spread.get('num_contracts', 0)
    if num_contracts is None:
        num_contracts = 0
    
    # Get max loss per contract
    max_loss_per_contract = best_spread.get('max_loss_per_contract')
    if max_loss_per_contract is None:
        # Calculate from max_loss_per_share if available
        max_loss_per_share = best_spread.get('max_loss')
        if max_loss_per_share is None or max_loss_per_share == 0:
            # Calculate from width and credit
            width = best_spread.get('width', 0)
            net_credit = best_spread.get('net_credit', 0)
            max_loss_per_share = width - net_credit
        max_loss_per_contract = max_loss_per_share * 100
    
    position_capital = max_loss_per_contract * num_contracts
    
    # Get calendar date in output timezone
    calendar_date = get_calendar_date(result['timestamp'], output_tz)
    
    return position_capital, calendar_date


def get_position_close_time(result: Dict, output_tz=None) -> Optional[datetime]:
    """Get the time when position closes (early exit or EOD).
    
    Args:
        result: Result dictionary from analyze_interval
        output_tz: Output timezone
    
    Returns:
        Close timestamp or None if position never closes
    """
    # Check for early exit (profit target hit)
    if result.get('profit_target_hit') is True:
        # Use exit_timestamp if available (from check_profit_target_hit)
        # For now, we'll use close_time_used which should be set
        close_time = result.get('close_time_used')
        if close_time is not None:
            return close_time
    
    # Check for force close hour
    close_time = result.get('close_time_used')
    if close_time is not None:
        return close_time
    
    # Otherwise, use EOD (end of trading day)
    # Get the date and set to 4:00 PM ET (market close)
    timestamp = result['timestamp']
    timestamp = normalize_timestamp(timestamp)
    
    if output_tz is not None:
        timestamp_local = timestamp.astimezone(output_tz)
        trading_date = timestamp_local.date()
    else:
        trading_date = timestamp.date() if hasattr(timestamp, 'date') else pd.to_datetime(timestamp).date()
    
    # Set to 4:00 PM ET
    et_tz = get_timezone("America/New_York")
    eod_et = get_eod_time(trading_date, et_tz)
    
    if output_tz:
        close_time = eod_et.astimezone(output_tz)
        return close_time
    
    return eod_et


def filter_results_by_capital_limit(
    results: List[Dict],
    max_live_capital: float,
    output_tz=None,
    logger: Optional[logging.Logger] = None
) -> List[Dict]:
    """Filter results based on daily capital limit, accounting for position lifecycle.
    
    Positions that close early free up capital for later positions.
    
    Args:
        results: List of result dictionaries
        max_live_capital: Maximum capital allowed per day
        output_tz: Output timezone
        logger: Optional logger
    
    Returns:
        Filtered list of results that fit within capital limits
    """
    if not results or max_live_capital is None:
        return results
    
    # Build timeline of events: position opens and closes
    events = []  # List of (timestamp, event_type, result, capital)
    
    for result in results:
        position_capital, calendar_date = calculate_position_capital(result, output_tz)
        open_time = result['timestamp']
        open_time = normalize_timestamp(open_time)
        
        if output_tz is not None:
            open_time = open_time.astimezone(output_tz)
        
        # Get close time
        close_time = get_position_close_time(result, output_tz)
        if close_time is None:
            # If we can't determine close time, assume EOD
            if hasattr(open_time, 'date'):
                trading_date = open_time.date()
            else:
                trading_date = pd.to_datetime(open_time).date()
            
            et_tz = get_timezone("America/New_York")
            eod_et = get_eod_time(trading_date, et_tz)
            
            if output_tz:
                close_time = eod_et.astimezone(output_tz)
            else:
                close_time = eod_et
        
        # Normalize close_time
        close_time = normalize_timestamp(close_time)
        if output_tz is not None:
            close_time = close_time.astimezone(output_tz)
        
        events.append((open_time, 'open', result, position_capital, calendar_date))
        events.append((close_time, 'close', result, position_capital, calendar_date))
    
    # Sort events chronologically
    events.sort(key=lambda x: x[0])
    
    # Process events chronologically, tracking available capital per day
    daily_available_capital = {}  # {date: available_capital}
    filtered_results = []
    opened_positions = set()  # Track which positions were actually opened (by result id)
    
    for event_time, event_type, result, position_capital, calendar_date in events:
        # Initialize available capital for this date if needed
        if calendar_date not in daily_available_capital:
            daily_available_capital[calendar_date] = max_live_capital
        
        if event_type == 'open':
            # Check if we have enough capital
            available = daily_available_capital[calendar_date]
            if position_capital <= available:
                # Open position
                daily_available_capital[calendar_date] -= position_capital
                filtered_results.append(result)
                opened_positions.add(id(result))
                
                if logger:
                    logger.debug(
                        f"Opened position at {event_time}: {position_capital:.2f} capital, "
                        f"Available for {calendar_date}: {daily_available_capital[calendar_date]:.2f}"
                    )
            else:
                if logger:
                    logger.debug(
                        f"Skipping position at {event_time}: "
                        f"Insufficient capital ({available:.2f} < {position_capital:.2f})"
                    )
        
        elif event_type == 'close':
            # Check if this position was actually opened
            if id(result) in opened_positions:
                # Close position - free up capital
                daily_available_capital[calendar_date] += position_capital
                
                if logger:
                    logger.debug(
                        f"Closed position at {event_time}: Freed {position_capital:.2f} capital, "
                        f"Available for {calendar_date}: {daily_available_capital[calendar_date]:.2f}"
                    )
    
    return filtered_results
