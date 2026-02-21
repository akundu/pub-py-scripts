"""
Data provider for fetching stock data from db_server.py.
"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class DataProvider(ABC):
    """Abstract base class for data providers."""
    
    @abstractmethod
    async def get_stock_data(self, symbol: str, start_date: str, end_date: str, interval: str) -> pd.DataFrame:
        """Get stock data for a symbol and date range."""
        pass
    
    @abstractmethod
    async def get_realtime_data(self, symbol: str, start_datetime: Optional[str] = None, 
                               end_datetime: Optional[str] = None, data_type: str = "quote") -> pd.DataFrame:
        """Get realtime data for a symbol."""
        pass


class DbServerProvider(DataProvider):
    """Data provider that connects to db_server.py on port 9002."""
    
    def __init__(self, host: str = "localhost", port: int = 9002, timeout: int = 30):
        self.base_url = f"http://{host}:{port}"
        self.timeout = timeout
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout))
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    async def _make_request(self, command: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Make a request to the db_server."""
        if not self.session:
            self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout))
        
        payload = {
            "command": command,
            "params": params
        }
        
        try:
            async with self.session.post(
                f"{self.base_url}/db_command",
                json=payload,
                headers={"Content-Type": "application/json"}
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"HTTP {response.status}: {error_text}")
                
                result = await response.json()
                if "error" in result:
                    raise Exception(f"Server error: {result['error']}")
                
                return result
                
        except asyncio.TimeoutError:
            raise Exception(f"Request timeout after {self.timeout} seconds")
        except Exception as e:
            raise Exception(f"Request failed: {str(e)}")
    
    async def get_stock_data(self, symbol: str, start_date: str, end_date: str, interval: str) -> pd.DataFrame:
        """Get stock data for a symbol and date range."""
        params = {
            "ticker": symbol,
            "start_date": start_date,
            "end_date": end_date,
            "interval": interval
        }
        
        result = await self._make_request("get_stock_data", params)
        
        if not result.get("data"):
            return pd.DataFrame()
        
        df = pd.DataFrame(result["data"])
        if df.empty:
            return df
        
        # Convert timestamp column to datetime
        if 'date' in df.columns:
            df['timestamp'] = pd.to_datetime(df['date'])
            df = df.drop('date', axis=1)
        elif 'datetime' in df.columns:
            df['timestamp'] = pd.to_datetime(df['datetime'])
            df = df.drop('datetime', axis=1)
        
        # Ensure required columns exist
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in df.columns:
                logger.warning(f"Missing column {col} for {symbol}")
                df[col] = 0.0
        
        # Set timestamp as index and sort
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)
        
        return df
    
    async def get_realtime_data(self, symbol: str, start_datetime: Optional[str] = None, 
                               end_datetime: Optional[str] = None, data_type: str = "quote") -> pd.DataFrame:
        """Get realtime data for a symbol."""
        params = {
            "ticker": symbol,
            "start_datetime": start_datetime,
            "end_datetime": end_datetime,
            "data_type": data_type
        }
        
        result = await self._make_request("get_realtime_data", params)
        
        if not result.get("data"):
            return pd.DataFrame()
        
        df = pd.DataFrame(result["data"])
        if df.empty:
            return df
        
        # Convert timestamp column to datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)
        
        return df
    
    async def get_daily(self, symbol: str, lookback_days: int = 90) -> pd.DataFrame:
        """Get daily data for a symbol with lookback period."""
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
        
        return await self.get_stock_data(symbol, start_date, end_date, "daily")
    
    async def get_hourly(self, symbol: str, lookback_days: int = 30) -> pd.DataFrame:
        """Get hourly data for a symbol with lookback period."""
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
        
        return await self.get_stock_data(symbol, start_date, end_date, "hourly")
    
    async def get_realtime_window(self, symbol: str, window_days: int = 7) -> pd.DataFrame:
        """Get realtime data for a symbol within a time window."""
        end_datetime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        start_datetime = (datetime.now() - timedelta(days=window_days)).strftime('%Y-%m-%d %H:%M:%S')
        
        return await self.get_realtime_data(symbol, start_datetime, end_datetime, "quote")
    
    async def test_connection(self) -> bool:
        """Test the connection to db_server."""
        try:
            result = await self._make_request("get_latest_price", {"ticker": "SPY"})
            return True
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False


class MockDataProvider(DataProvider):
    """Mock data provider for testing purposes."""
    
    def __init__(self):
        # Generate synthetic data for testing
        dates = pd.date_range(start='2020-01-01', end='2024-12-31', freq='D')
        np.random.seed(42)
        
        # Generate random price data
        base_price = 100.0
        returns = np.random.normal(0.001, 0.02, len(dates))  # Daily returns with 0.1% mean, 2% std
        
        prices = [base_price]
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        self.mock_data = pd.DataFrame({
            'timestamp': dates,
            'open': [p * (1 + np.random.normal(0, 0.005)) for p in prices],
            'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
            'close': prices,
            'volume': np.random.randint(1000000, 10000000, len(dates))
        }).set_index('timestamp')
    
    async def get_stock_data(self, symbol: str, start_date: str, end_date: str, interval: str) -> pd.DataFrame:
        """Get mock stock data."""
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        
        mask = (self.mock_data.index >= start) & (self.mock_data.index <= end)
        return self.mock_data[mask].copy()
    
    async def get_realtime_data(self, symbol: str, start_datetime: Optional[str] = None, 
                               end_datetime: Optional[str] = None, data_type: str = "quote") -> pd.DataFrame:
        """Get mock realtime data."""
        # For testing, return a subset of daily data as if it were realtime
        return await self.get_stock_data(symbol, "2024-01-01", "2024-01-31", "daily")
    
    async def get_daily(self, symbol: str, lookback_days: int = 90) -> pd.DataFrame:
        """Get mock daily data."""
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
        return await self.get_stock_data(symbol, start_date, end_date, "daily")
    
    async def get_hourly(self, symbol: str, lookback_days: int = 30) -> pd.DataFrame:
        """Get mock hourly data."""
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
        return await self.get_stock_data(symbol, start_date, end_date, "hourly")
    
    async def get_realtime_window(self, symbol: str, window_days: int = 7) -> pd.DataFrame:
        """Get mock realtime data."""
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=window_days)).strftime('%Y-%m-%d')
        return await self.get_stock_data(symbol, start_date, end_date, "daily")
    
    async def test_connection(self) -> bool:
        """Mock connection test always succeeds."""
        return True
