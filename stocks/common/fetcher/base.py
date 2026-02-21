"""
Abstract base class for data fetchers.

Provides common functionality for date validation, normalization,
and data formatting across different data sources.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict, Any
import pandas as pd
import logging

logger = logging.getLogger(__name__)


@dataclass
class FetchResult:
    """Result of a data fetch operation."""
    data: pd.DataFrame
    source: str
    symbol: str
    timeframe: str
    start_date: str
    end_date: str
    records_fetched: int
    success: bool
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class AbstractDataFetcher(ABC):
    """
    Abstract base class for financial data fetchers.
    
    Provides common functionality for:
    - Date validation and normalization
    - Timeframe handling
    - Data formatting
    - Error handling
    
    Subclasses must implement:
    - fetch_historical_data()
    - fetch_current_price()
    """
    
    def __init__(self, name: str, log_level: str = "INFO"):
        self.name = name
        self.log_level = log_level
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging configuration."""
        numeric_level = getattr(logging, self.log_level.upper(), logging.INFO)
        logger.setLevel(numeric_level)
    
    @property
    @abstractmethod
    def supported_timeframes(self) -> List[str]:
        """List of supported timeframes (e.g., ['daily', 'hourly'])."""
        pass
    
    @property
    @abstractmethod
    def max_historical_days(self) -> Dict[str, Optional[int]]:
        """
        Maximum historical days for each timeframe.
        
        Returns:
            Dict mapping timeframe to max days (None = unlimited)
            Example: {'daily': None, 'hourly': 729}
        """
        pass
    
    def validate_timeframe(self, timeframe: str) -> None:
        """Validate that timeframe is supported."""
        if timeframe not in self.supported_timeframes:
            raise ValueError(
                f"{self.name} does not support timeframe '{timeframe}'. "
                f"Supported: {self.supported_timeframes}"
            )
    
    def normalize_dates(
        self,
        start_date: str,
        end_date: str,
        timeframe: str
    ) -> tuple[datetime, datetime]:
        """
        Normalize and validate date range.
        
        Args:
            start_date: Start date string (ISO or YYYY-MM-DD)
            end_date: End date string (ISO or YYYY-MM-DD)
            timeframe: Data timeframe
            
        Returns:
            Tuple of (start_dt, end_dt) as timezone-aware datetimes
            
        Raises:
            ValueError: If dates are invalid
        """
        # Parse dates
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        # Get current time
        now = datetime.now(timezone.utc)
        today = now.replace(hour=0, minute=0, second=0, microsecond=0)
        
        # Add timezone if naive
        if start_dt.tzinfo is None:
            start_dt = start_dt.replace(tzinfo=timezone.utc)
        if end_dt.tzinfo is None:
            end_dt = end_dt.replace(tzinfo=timezone.utc)
        
        # Normalize to start of day
        start_dt_normalized = start_dt.replace(hour=0, minute=0, second=0, microsecond=0)
        end_dt_normalized = end_dt.replace(hour=0, minute=0, second=0, microsecond=0)
        
        # Validate start date is not in future
        if start_dt_normalized > today:
            raise ValueError(
                f"Start date {start_date} is in the future. "
                f"Current date is {today.strftime('%Y-%m-%d')}"
            )
        
        # Adjust end date if in future
        if end_dt_normalized > today:
            days_diff = (end_dt_normalized - today).days
            if days_diff > 1:
                logger.warning(
                    f"End date {end_date} is {days_diff} days in the future. "
                    f"Adjusting to today ({today.strftime('%Y-%m-%d')})"
                )
            end_dt_normalized = today
        
        # Apply timeframe-specific limits
        max_days = self.max_historical_days.get(timeframe)
        if max_days is not None:
            max_allowed_start = today - timedelta(days=max_days)
            
            if start_dt_normalized < max_allowed_start:
                original_start = start_dt_normalized
                start_dt_normalized = max_allowed_start
                days_requested = (end_dt_normalized - original_start).days
                logger.warning(
                    f"{self.name} {timeframe} data is limited to the last {max_days} days. "
                    f"Adjusting start date from {original_start.strftime('%Y-%m-%d')} "
                    f"to {start_dt_normalized.strftime('%Y-%m-%d')} "
                    f"(requested {days_requested} days)"
                )
            
            # For hourly data, always use today as end date (Yahoo Finance requirement)
            if timeframe == "hourly" and end_dt_normalized != today:
                logger.info(
                    f"For hourly data, adjusting end date to today "
                    f"({today.strftime('%Y-%m-%d')}) for {self.name}"
                )
                end_dt_normalized = today
        
        return start_dt_normalized, end_dt_normalized
    
    def format_dataframe(
        self,
        data: pd.DataFrame,
        timeframe: str,
        ensure_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Format DataFrame to standard format.
        
        Args:
            data: Raw DataFrame
            timeframe: Timeframe (for index naming)
            ensure_columns: Required columns to validate
            
        Returns:
            Formatted DataFrame with lowercase columns and proper index
        """
        if data.empty:
            return data
        
        # Lowercase column names
        data.columns = [col.lower() for col in data.columns]
        
        # Ensure DatetimeIndex
        if not isinstance(data.index, pd.DatetimeIndex):
            data.index = pd.to_datetime(data.index)
        
        # Set index name based on timeframe
        if timeframe == "daily":
            data.index.name = 'date'
        else:
            data.index.name = 'datetime'
        
        # Ensure timezone is UTC
        if data.index.tz is None:
            data.index = data.index.tz_localize('UTC')
        else:
            data.index = data.index.tz_convert('UTC')
        
        # Validate required columns
        if ensure_columns:
            missing = [col for col in ensure_columns if col not in data.columns]
            if missing:
                logger.warning(f"Missing columns in data: {missing}")
        
        return data
    
    @abstractmethod
    async def fetch_historical_data(
        self,
        symbol: str,
        timeframe: str,
        start_date: str,
        end_date: str,
        **kwargs
    ) -> FetchResult:
        """
        Fetch historical OHLCV data.
        
        Args:
            symbol: Ticker symbol
            timeframe: 'daily' or 'hourly'
            start_date: Start date (ISO or YYYY-MM-DD)
            end_date: End date (ISO or YYYY-MM-DD)
            **kwargs: Additional fetcher-specific parameters
            
        Returns:
            FetchResult with data and metadata
        """
        pass
    
    @abstractmethod
    async def fetch_current_price(
        self,
        symbol: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Fetch current/latest price.
        
        Args:
            symbol: Ticker symbol
            **kwargs: Additional fetcher-specific parameters
            
        Returns:
            Dict with price, timestamp, and other info
        """
        pass
    
    def create_error_result(
        self,
        symbol: str,
        timeframe: str,
        start_date: str,
        end_date: str,
        error: str
    ) -> FetchResult:
        """Create an error FetchResult."""
        return FetchResult(
            data=pd.DataFrame(),
            source=self.name,
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            records_fetched=0,
            success=False,
            error=error
        )
