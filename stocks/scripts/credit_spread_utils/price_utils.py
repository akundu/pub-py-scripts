"""
Price fetching utilities for credit spread analysis.

Consolidates price fetching functions from QuestDB, with CSV-based fallback
when the database is unavailable.
"""

from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple, Dict, Any, List
import logging
import os
from pathlib import Path

import pandas as pd

from .timezone_utils import (
    get_timezone,
    normalize_timestamp,
    get_eod_time,
    get_previous_trading_day,
)


# Module-level cache for DB price queries (avoids repeated DB hits in grid search)
_db_price_cache: Dict[str, Any] = {}

# Default equities directory (relative to the stocks project root)
_DEFAULT_EQUITIES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'equities_output')


def _get_close_from_csv(
    ticker: str,
    target_date,
    equities_dir: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> Optional[float]:
    """
    Get closing price from equities CSV files.

    Tries multiple ticker formats: I:{ticker}, {ticker}, and the ticker as-is.

    Args:
        ticker: Ticker symbol (e.g., 'NDX', 'SPX', 'I:NDX')
        target_date: Date to get close price for (date or datetime)
        equities_dir: Directory containing equities data
        logger: Optional logger

    Returns:
        Closing price or None if not found
    """
    if equities_dir is None:
        equities_dir = _DEFAULT_EQUITIES_DIR

    base_dir = Path(equities_dir)
    if not base_dir.exists():
        return None

    if hasattr(target_date, 'date'):
        target_date = target_date.date()

    date_str = target_date.strftime('%Y-%m-%d')

    # Try multiple ticker formats for directory and filename
    ticker_variants = [ticker]
    if not ticker.startswith('I:'):
        ticker_variants.append(f'I:{ticker}')
    else:
        ticker_variants.append(ticker[2:])  # strip I: prefix

    for tv in ticker_variants:
        ticker_dir = base_dir / tv
        if not ticker_dir.exists():
            continue

        filename = f"{tv}_equities_{date_str}.csv"
        filepath = ticker_dir / filename
        if not filepath.exists():
            continue

        try:
            df = pd.read_csv(filepath)
            if 'close' in df.columns and len(df) > 0:
                # Get last row's close (EOD)
                close_val = float(df['close'].iloc[-1])
                if logger:
                    logger.debug(f"CSV fallback: {ticker} close=${close_val:.2f} for {date_str}")
                return close_val
        except Exception as e:
            if logger:
                logger.debug(f"CSV fallback error reading {filepath}: {e}")
            continue

    return None


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
            logger.debug(f"DEBUG get_current_day_close_price: No data found in DB for {ticker} on {trading_day}, trying CSV fallback")
        csv_close = _get_close_from_csv(ticker, trading_day, logger=logger)
        if csv_close is not None:
            result = (csv_close, trading_day)
            _db_price_cache[cache_key] = result
            return result
        _db_price_cache[cache_key] = None
        return None
    except Exception as e:
        if logger:
            logger.debug(f"DB unavailable for current day close {ticker}: {e}, trying CSV fallback")
        try:
            et_tz = get_timezone("America/New_York")
            ref_normalized = normalize_timestamp(reference_date)
            trading_day = ref_normalized.astimezone(et_tz).date()
            csv_close = _get_close_from_csv(ticker, trading_day, logger=logger)
            if csv_close is not None:
                result = (csv_close, trading_day)
                _db_price_cache[cache_key] = result
                return result
        except Exception as csv_e:
            if logger:
                logger.debug(f"CSV fallback also failed for current day close {ticker}: {csv_e}")
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
            logger.debug(f"DEBUG get_previous_close_price: No data found in DB for {ticker}, trying CSV fallback")
        # CSV fallback when DB has no data
        csv_close = _get_close_from_csv(ticker, prev_trading_day, logger=logger)
        if csv_close is not None:
            prev_date = prev_trading_day.date() if hasattr(prev_trading_day, 'date') else prev_trading_day
            result = (csv_close, prev_date)
            _db_price_cache[cache_key] = result
            return result
        _db_price_cache[cache_key] = None
        return None
    except Exception as e:
        if logger:
            logger.debug(f"DB unavailable for {ticker}: {e}, trying CSV fallback")
        # CSV fallback when DB is unavailable
        try:
            prev_trading_day = get_previous_trading_day(reference_date)
            csv_close = _get_close_from_csv(ticker, prev_trading_day, logger=logger)
            if csv_close is not None:
                prev_date = prev_trading_day.date() if hasattr(prev_trading_day, 'date') else prev_trading_day
                result = (csv_close, prev_date)
                _db_price_cache[cache_key] = result
                return result
        except Exception as csv_e:
            if logger:
                logger.debug(f"CSV fallback also failed for {ticker}: {csv_e}")
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


async def get_historical_price_patterns(
    db,
    ticker: str,
    lookback_days: int = 365,
    hours: Optional[List[int]] = None,
    logger: Optional[logging.Logger] = None
) -> pd.DataFrame:
    """
    Query hourly prices joined with daily close and VIX1D data.

    Fetches historical data for close prediction model training.
    Returns data for each hour of each trading day within the lookback period.

    Args:
        db: Database connection (StockQuestDB instance)
        ticker: Stock ticker (e.g., 'I:NDX', 'I:SPX')
        lookback_days: Number of days to look back (default 365)
        hours: List of hours to include (ET, 9-15). Default: [9, 10, 11, 12, 13, 14, 15]
        logger: Optional logger

    Returns:
        DataFrame with columns:
        - date: trading date (date object)
        - hour_et: hour in Eastern Time (9-15)
        - hour_price: price at that hour (close of that hourly bar)
        - day_open: opening price for the day
        - day_close: closing price for the day
        - prev_close: previous day's closing price
        - vix1d: VIX1D value for that day
        - day_of_week: day of week (0=Monday, 4=Friday)
    """
    if hours is None:
        hours = [9, 10, 11, 12, 13, 14, 15]

    if logger is None:
        logger = logging.getLogger(__name__)

    logger.info(f"Fetching historical price patterns for {ticker}, lookback={lookback_days} days")

    et_tz = get_timezone("America/New_York")
    end_date = datetime.now(et_tz)
    start_date = end_date - timedelta(days=lookback_days)

    # Convert to UTC for queries
    start_utc = start_date.astimezone(timezone.utc).replace(tzinfo=None)
    end_utc = end_date.astimezone(timezone.utc).replace(tzinfo=None)

    try:
        async with db.connection.get_connection() as conn:
            # Step 1: Get daily prices for the ticker
            daily_query = """
                SELECT
                    date,
                    open,
                    close
                FROM daily_prices
                WHERE ticker = $1
                  AND date >= $2
                  AND date < $3
                ORDER BY date ASC
            """
            daily_rows = await conn.fetch(daily_query, ticker, start_utc, end_utc)

            if not daily_rows:
                logger.warning(f"No daily data found for {ticker}")
                return pd.DataFrame()

            # Convert to DataFrame
            daily_df = pd.DataFrame([dict(row) for row in daily_rows])
            daily_df['date'] = pd.to_datetime(daily_df['date']).dt.date
            daily_df['open'] = daily_df['open'].astype(float)
            daily_df['close'] = daily_df['close'].astype(float)

            # Calculate previous close
            daily_df['prev_close'] = daily_df['close'].shift(1)

            # Step 2: Get hourly prices for the ticker
            hourly_query = """
                SELECT
                    datetime,
                    close
                FROM hourly_prices
                WHERE ticker = $1
                  AND datetime >= $2
                  AND datetime < $3
                ORDER BY datetime ASC
            """
            hourly_rows = await conn.fetch(hourly_query, ticker, start_utc, end_utc)

            if not hourly_rows:
                logger.warning(f"No hourly data found for {ticker}")
                return pd.DataFrame()

            # Convert to DataFrame
            hourly_df = pd.DataFrame([dict(row) for row in hourly_rows])
            hourly_df['datetime'] = pd.to_datetime(hourly_df['datetime'])

            # Convert to ET and extract date/hour
            hourly_df['datetime_et'] = hourly_df['datetime'].dt.tz_localize('UTC').dt.tz_convert(et_tz)
            hourly_df['date'] = hourly_df['datetime_et'].dt.date
            hourly_df['hour_et'] = hourly_df['datetime_et'].dt.hour
            hourly_df['hour_price'] = hourly_df['close'].astype(float)

            # Filter to requested hours
            hourly_df = hourly_df[hourly_df['hour_et'].isin(hours)]

            # Step 3: Get VIX1D data
            vix_query = """
                SELECT
                    date,
                    close as vix1d
                FROM daily_prices
                WHERE ticker = 'I:VIX1D'
                  AND date >= $1
                  AND date < $2
                ORDER BY date ASC
            """
            vix_rows = await conn.fetch(vix_query, start_utc, end_utc)

            vix_df = pd.DataFrame()
            if vix_rows:
                vix_df = pd.DataFrame([dict(row) for row in vix_rows])
                vix_df['date'] = pd.to_datetime(vix_df['date']).dt.date
                vix_df['vix1d'] = vix_df['vix1d'].astype(float)
            else:
                logger.warning("No VIX1D data found, predictions will use default VIX values")

            # Step 4: Merge all data
            # Start with hourly data
            result_df = hourly_df[['date', 'hour_et', 'hour_price']].copy()

            # Merge with daily data
            result_df = result_df.merge(
                daily_df[['date', 'open', 'close', 'prev_close']].rename(
                    columns={'open': 'day_open', 'close': 'day_close'}
                ),
                on='date',
                how='inner'
            )

            # Merge with VIX data
            if not vix_df.empty:
                result_df = result_df.merge(
                    vix_df[['date', 'vix1d']],
                    on='date',
                    how='left'
                )
            else:
                result_df['vix1d'] = None

            # Add day of week
            result_df['day_of_week'] = pd.to_datetime(result_df['date']).dt.dayofweek

            # Drop rows with missing essential data
            result_df = result_df.dropna(subset=['hour_price', 'day_close', 'prev_close'])

            # Sort by date and hour
            result_df = result_df.sort_values(['date', 'hour_et']).reset_index(drop=True)

            logger.info(f"Retrieved {len(result_df)} historical price records for {ticker}")

            return result_df

    except Exception as e:
        logger.error(f"Error fetching historical price patterns for {ticker}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return pd.DataFrame()


async def get_current_vix1d(
    db,
    logger: Optional[logging.Logger] = None
) -> Optional[float]:
    """
    Get the most recent VIX1D value.

    Args:
        db: Database connection
        logger: Optional logger

    Returns:
        Most recent VIX1D close value or None
    """
    try:
        async with db.connection.get_connection() as conn:
            rows = await conn.fetch(
                """
                SELECT close
                FROM daily_prices
                WHERE ticker = 'I:VIX1D'
                ORDER BY date DESC
                LIMIT 1
                """
            )

            if rows:
                return float(rows[0]['close'])

        return None
    except Exception as e:
        if logger:
            logger.error(f"Error getting VIX1D: {e}")
        return None


async def get_intraday_high_low(
    db,
    ticker: str,
    reference_date: datetime,
    logger: Optional[logging.Logger] = None
) -> Optional[Tuple[float, float]]:
    """
    Get the high and low prices for the current trading day so far.

    Args:
        db: Database connection
        ticker: Stock ticker
        reference_date: Reference datetime (timezone-aware)
        logger: Optional logger

    Returns:
        Tuple of (day_high, day_low) or None if not found
    """
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
        current_utc = reference_date.astimezone(timezone.utc).replace(tzinfo=None)

        async with db.connection.get_connection() as conn:
            # Query hourly_prices for high/low
            rows = await conn.fetch(
                """
                SELECT MAX(high) as day_high, MIN(low) as day_low
                FROM hourly_prices
                WHERE ticker = $1
                  AND datetime >= $2
                  AND datetime <= $3
                """,
                ticker, day_start_utc, current_utc
            )

            if rows and rows[0]['day_high'] is not None:
                return (float(rows[0]['day_high']), float(rows[0]['day_low']))

        return None
    except Exception as e:
        if logger:
            logger.error(f"Error getting intraday high/low for {ticker}: {e}")
        return None
