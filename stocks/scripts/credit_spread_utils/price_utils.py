"""
Price fetching utilities for credit spread analysis.

Consolidates price fetching functions from QuestDB.
"""

from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple, Dict, Any
import logging

from .timezone_utils import (
    get_timezone,
    normalize_timestamp,
    get_eod_time,
    get_previous_trading_day,
)


# Module-level cache for DB price queries (avoids repeated DB hits in grid search)
_db_price_cache: Dict[str, Any] = {}


def clear_price_cache():
    """Clear the price cache."""
    global _db_price_cache
    _db_price_cache.clear()


async def get_current_day_close_price(
    db,
    ticker: str,
    reference_date: datetime,
    logger: Optional[logging.Logger] = None
) -> Optional[Tuple[float, datetime.date]]:
    """Get the closing price for the trading day of reference_date.

    Returns:
        Tuple of (close_price, trading_day_date) or None if not found
    """
    # Check memo cache
    cache_key = f"current:{ticker}:{reference_date.date()}"
    if cache_key in _db_price_cache:
        return _db_price_cache[cache_key]

    try:
        et_tz = get_timezone("America/New_York")
        reference_date = normalize_timestamp(reference_date)
        
        # Convert to ET
        date_et = reference_date.astimezone(et_tz)
        trading_day = date_et.date()
        
        # Create start of day in ET, then convert to UTC
        from datetime import datetime as dt_class
        try:
            # Try zoneinfo approach
            day_start_et = dt_class(trading_day.year, trading_day.month, trading_day.day, tzinfo=et_tz)
        except (TypeError, ValueError):
            # Fallback to pytz localize
            try:
                day_start_et = et_tz.localize(dt_class(trading_day.year, trading_day.month, trading_day.day))
            except AttributeError:
                day_start_et = dt_class(trading_day.year, trading_day.month, trading_day.day, tzinfo=et_tz)
        
        day_start_utc = day_start_et.astimezone(timezone.utc).replace(tzinfo=None)
        day_end_utc = day_start_utc + timedelta(days=1)
        
        if logger:
            logger.debug(f"DEBUG get_current_day_close_price: ticker={ticker}, trading_day={trading_day}, day_start_utc={day_start_utc}, day_end_utc={day_end_utc}")
        
        async with db.connection.get_connection() as conn:
            # Get close price for the trading day
            rows = await conn.fetch(
                "SELECT date, close FROM daily_prices WHERE ticker = $1 AND date >= $2 AND date < $3 ORDER BY date DESC LIMIT 1",
                ticker, day_start_utc, day_end_utc
            )
            
            if logger:
                logger.debug(f"DEBUG get_current_day_close_price: Found {len(rows)} rows")
            
            if rows:
                close_price = float(rows[0]['close'])
                row_date = rows[0]['date']
                if isinstance(row_date, datetime):
                    trading_date = row_date.date()
                else:
                    trading_date = trading_day
                if logger:
                    logger.debug(f"DEBUG get_current_day_close_price: Returning close=${close_price:.2f} for date={trading_date}")
                result = (close_price, trading_date)
                _db_price_cache[cache_key] = result
                return result

        if logger:
            logger.debug(f"DEBUG get_current_day_close_price: No data found for {ticker} on {trading_day}")
        _db_price_cache[cache_key] = None
        return None
    except Exception as e:
        error_msg = f"Error getting current day close for {ticker}: {e}"
        if logger:
            logger.error(error_msg)
        else:
            logging.error(error_msg)
        _db_price_cache[cache_key] = None
        return None


async def get_previous_close_price(
    db,
    ticker: str,
    reference_date: datetime,
    logger: Optional[logging.Logger] = None
) -> Optional[Tuple[float, datetime.date]]:
    """Get the closing price for the previous trading day relative to reference_date.

    Returns:
        Tuple of (close_price, prev_trading_day_date) or None if not found
    """
    # Check memo cache
    cache_key = f"prev:{ticker}:{reference_date.date()}"
    if cache_key in _db_price_cache:
        return _db_price_cache[cache_key]

    try:
        # Get previous trading day
        prev_trading_day = get_previous_trading_day(reference_date)
        
        # Query daily_prices for that date
        # QuestDB stores dates as UTC timestamps
        # Create start of day in ET, then convert to UTC
        et_tz = get_timezone("America/New_York")
        
        from datetime import datetime as dt_class
        try:
            prev_day_start_et = dt_class(prev_trading_day.year, prev_trading_day.month, prev_trading_day.day, tzinfo=et_tz)
        except (TypeError, ValueError):
            try:
                prev_day_start_et = et_tz.localize(dt_class(prev_trading_day.year, prev_trading_day.month, prev_trading_day.day))
            except AttributeError:
                prev_day_start_et = dt_class(prev_trading_day.year, prev_trading_day.month, prev_trading_day.day, tzinfo=et_tz)
        
        prev_day_start_utc = prev_day_start_et.astimezone(timezone.utc).replace(tzinfo=None)
        prev_day_end_utc = prev_day_start_utc + timedelta(days=1)
        
        if logger:
            logger.debug(f"DEBUG get_previous_close_price: ticker={ticker}, prev_trading_day={prev_trading_day}, prev_day_start_utc={prev_day_start_utc}, prev_day_end_utc={prev_day_end_utc}")
        
        async with db.connection.get_connection() as conn:
            # First try to get exact match for previous trading day
            rows = await conn.fetch(
                "SELECT date, close FROM daily_prices WHERE ticker = $1 AND date >= $2 AND date < $3 ORDER BY date DESC LIMIT 1",
                ticker, prev_day_start_utc, prev_day_end_utc
            )
            
            if logger:
                logger.debug(f"DEBUG get_previous_close_price: First query found {len(rows)} rows")
            
            if rows:
                close_price = float(rows[0]['close'])
                # Extract date from the row
                row_date = rows[0]['date']
                if isinstance(row_date, datetime):
                    prev_date = row_date.date()
                else:
                    prev_date = prev_trading_day.date()
                if logger:
                    logger.debug(f"DEBUG get_previous_close_price: Returning close=${close_price:.2f} for date={prev_date}")
                result = (close_price, prev_date)
                _db_price_cache[cache_key] = result
                return result

            # If not found, get the most recent close before the current trading day
            # Calculate the start of the current trading day to exclude it
            ref_date_for_cutoff = normalize_timestamp(reference_date)
            
            # Convert to ET to get the trading day
            date_et = ref_date_for_cutoff.astimezone(et_tz)
            trading_day = date_et.date()
            
            # Calculate start of current trading day in UTC
            try:
                day_start_et = dt_class(trading_day.year, trading_day.month, trading_day.day, tzinfo=et_tz)
            except (TypeError, ValueError):
                try:
                    day_start_et = et_tz.localize(dt_class(trading_day.year, trading_day.month, trading_day.day))
                except AttributeError:
                    day_start_et = dt_class(trading_day.year, trading_day.month, trading_day.day, tzinfo=et_tz)
            day_start_utc = day_start_et.astimezone(timezone.utc).replace(tzinfo=None)
            
            if logger:
                logger.debug(f"DEBUG get_previous_close_price: Fallback query - trading_day={trading_day}, day_start_utc={day_start_utc}")
            
            # Get most recent close before the start of current trading day
            # This ensures we never get the current day's close
            rows = await conn.fetch(
                "SELECT date, close FROM daily_prices WHERE ticker = $1 AND date < $2 ORDER BY date DESC LIMIT 1",
                ticker, day_start_utc
            )
            
            if logger:
                logger.debug(f"DEBUG get_previous_close_price: Fallback query found {len(rows)} rows")
            
            if rows:
                close_price = float(rows[0]['close'])
                row_date = rows[0]['date']
                if isinstance(row_date, datetime):
                    prev_date = row_date.date()
                else:
                    prev_date = prev_trading_day.date()
                if logger:
                    logger.debug(f"DEBUG get_previous_close_price: Fallback returning close=${close_price:.2f} for date={prev_date}")
                result = (close_price, prev_date)
                _db_price_cache[cache_key] = result
                return result

        if logger:
            logger.debug(f"DEBUG get_previous_close_price: No data found for {ticker}")
        _db_price_cache[cache_key] = None
        return None
    except Exception as e:
        error_msg = f"Error getting previous close for {ticker}: {e}"
        if logger:
            logger.error(error_msg)
        else:
            logging.error(error_msg)
        _db_price_cache[cache_key] = None
        return None


async def get_previous_open_price(
    db,
    ticker: str,
    reference_date: datetime,
    logger: Optional[logging.Logger] = None
) -> Optional[float]:
    """Get the opening price for the previous trading day relative to reference_date."""
    try:
        prev_close_result = await get_previous_close_price(db, ticker, reference_date, logger)
        if prev_close_result is None:
            return None
        
        prev_close, prev_date = prev_close_result
        
        # Get the open price for the same date
        et_tz = get_timezone("America/New_York")
        
        from datetime import datetime as dt_class
        try:
            day_start_et = dt_class(prev_date.year, prev_date.month, prev_date.day, tzinfo=et_tz)
        except (TypeError, ValueError):
            try:
                day_start_et = et_tz.localize(dt_class(prev_date.year, prev_date.month, prev_date.day))
            except AttributeError:
                day_start_et = dt_class(prev_date.year, prev_date.month, prev_date.day, tzinfo=et_tz)
        
        day_start_utc = day_start_et.astimezone(timezone.utc).replace(tzinfo=None)
        day_end_utc = day_start_utc + timedelta(days=1)
        
        async with db.connection.get_connection() as conn:
            rows = await conn.fetch(
                "SELECT open FROM daily_prices WHERE ticker = $1 AND date >= $2 AND date < $3 ORDER BY date DESC LIMIT 1",
                ticker, day_start_utc, day_end_utc
            )
            
            if rows and rows[0]['open'] is not None:
                return float(rows[0]['open'])
        
        return None
    except Exception as e:
        if logger:
            logger.debug(f"DEBUG get_previous_open_price: Error - {e}")
        return None


async def get_current_day_open_price(
    db,
    ticker: str,
    reference_date: datetime,
    logger: Optional[logging.Logger] = None
) -> Optional[float]:
    """Get the opening price for the trading day of reference_date."""
    try:
        et_tz = get_timezone("America/New_York")
        reference_date = normalize_timestamp(reference_date)
        
        # Convert to ET
        date_et = reference_date.astimezone(et_tz)
        trading_day = date_et.date()
        
        # Create start of day in ET, then convert to UTC
        from datetime import datetime as dt_class
        try:
            day_start_et = dt_class(trading_day.year, trading_day.month, trading_day.day, tzinfo=et_tz)
        except (TypeError, ValueError):
            try:
                day_start_et = et_tz.localize(dt_class(trading_day.year, trading_day.month, trading_day.day))
            except AttributeError:
                day_start_et = dt_class(trading_day.year, trading_day.month, trading_day.day, tzinfo=et_tz)
        
        day_start_utc = day_start_et.astimezone(timezone.utc).replace(tzinfo=None)
        day_end_utc = day_start_utc + timedelta(days=1)
        
        async with db.connection.get_connection() as conn:
            rows = await conn.fetch(
                "SELECT open FROM daily_prices WHERE ticker = $1 AND date >= $2 AND date < $3 ORDER BY date DESC LIMIT 1",
                ticker, day_start_utc, day_end_utc
            )
            
            if rows and rows[0]['open'] is not None:
                return float(rows[0]['open'])
        
        return None
    except Exception as e:
        if logger:
            logger.debug(f"DEBUG get_current_day_open_price: Error - {e}")
        return None


async def get_price_at_time(
    db,
    ticker: str,
    target_timestamp: datetime,
    logger: Optional[logging.Logger] = None
) -> Optional[float]:
    """Get price at a specific timestamp (from hourly_prices table).

    Args:
        db: Database connection
        ticker: Stock ticker
        target_timestamp: Target timestamp (timezone-aware)
        logger: Optional logger

    Returns:
        Price at that time or None if not found
    """
    try:
        et_tz = get_timezone("America/New_York")
        target_timestamp = normalize_timestamp(target_timestamp)
        
        # Convert target to ET
        target_et = target_timestamp.astimezone(et_tz)
        trading_day = target_et.date()
        
        # Get start and end of trading day in ET
        from datetime import datetime as dt_class
        try:
            day_start_et = et_tz.localize(dt_class(
                trading_day.year, trading_day.month, trading_day.day, 9, 30, 0
            ))
        except AttributeError:
            day_start_et = dt_class(
                trading_day.year, trading_day.month, trading_day.day, 9, 30, 0,
                tzinfo=et_tz
            )
        
        day_end_et = get_eod_time(trading_day, et_tz)
        day_start_utc = day_start_et.astimezone(timezone.utc).replace(tzinfo=None)
        day_end_utc = day_end_et.astimezone(timezone.utc).replace(tzinfo=None)
        target_utc = target_timestamp.astimezone(timezone.utc).replace(tzinfo=None)
        
        # Query hourly_prices for the closest time
        async with db.connection.get_connection() as conn:
            query = """
                SELECT datetime, close 
                FROM hourly_prices 
                WHERE ticker = $1 
                  AND datetime >= $2 
                  AND datetime <= $3
                  AND datetime <= $4
                ORDER BY ABS(EXTRACT(EPOCH FROM (datetime - $4))) ASC
                LIMIT 1
            """
            rows = await conn.fetch(query, ticker, day_start_utc, day_end_utc, target_utc)
            
            if rows:
                return float(rows[0]['close'])
        
        return None
    except Exception as e:
        if logger:
            logger.error(f"Error getting price at time for {ticker}: {e}")
        return None
