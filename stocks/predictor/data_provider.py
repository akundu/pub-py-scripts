"""
Data provider for connecting to db_server.py on port 9002.

This module provides a clean interface to fetch stock data from the database server.
"""

import asyncio
import aiohttp
import pandas as pd
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class DbServerProvider:
    """Data provider that connects to db_server.py on port 9002."""
    
    def __init__(self, host: str = "localhost", port: int = 9002, timeout: float = 30.0):
        """
        Initialize the database server provider.
        
        Args:
            host: Database server host
            port: Database server port  
            timeout: Request timeout in seconds
        """
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}"
        self.timeout = timeout
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.timeout)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    async def _make_request(self, command: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a request to the db_server.
        
        Args:
            command: Database command to execute
            params: Command parameters
            
        Returns:
            Response data from the server
            
        Raises:
            Exception: If request fails or server returns error
        """
        if not self.session:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            )
        
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
    
    async def get_daily(self, symbol: str, start_date: Optional[str] = None, 
                       end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Get daily stock data for a symbol.
        
        Args:
            symbol: Stock symbol
            start_date: Start date in YYYY-MM-DD format (optional)
            end_date: End date in YYYY-MM-DD format (optional)
            
        Returns:
            DataFrame with OHLCV data
        """
        # Set default dates if not provided
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        
        result = await self._make_request("get_stock_data", {
            "ticker": symbol,
            "start_date": start_date,
            "end_date": end_date,
            "interval": "daily"
        })
        
        if not result.get("data"):
            logger.warning(f"No daily data found for {symbol}")
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(result["data"])
        
        # Ensure we have the required columns
        required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.warning(f"Missing columns for {symbol}: {missing_cols}")
            return pd.DataFrame()
        
        # Convert date column to datetime and set as index
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        # Ensure numeric columns are properly typed
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Sort by date
        df.sort_index(inplace=True)
        
        # Remove any rows with NaN values
        df.dropna(inplace=True)
        
        logger.info(f"Retrieved {len(df)} daily records for {symbol}")
        return df
    
    async def get_hourly(self, symbol: str, start_date: Optional[str] = None,
                        end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Get hourly stock data for a symbol.
        
        Args:
            symbol: Stock symbol
            start_date: Start date in YYYY-MM-DD format (optional)
            end_date: End date in YYYY-MM-DD format (optional)
            
        Returns:
            DataFrame with OHLCV data
        """
        # Set default dates if not provided
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        
        result = await self._make_request("get_stock_data", {
            "ticker": symbol,
            "start_date": start_date,
            "end_date": end_date,
            "interval": "hourly"
        })
        
        if not result.get("data"):
            logger.warning(f"No hourly data found for {symbol}")
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(result["data"])
        
        # Ensure we have the required columns
        required_cols = ['datetime', 'open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.warning(f"Missing columns for {symbol}: {missing_cols}")
            return pd.DataFrame()
        
        # Convert datetime column to datetime and set as index
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)
        
        # Ensure numeric columns are properly typed
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Sort by datetime
        df.sort_index(inplace=True)
        
        # Remove any rows with NaN values
        df.dropna(inplace=True)
        
        logger.info(f"Retrieved {len(df)} hourly records for {symbol}")
        return df
    
    async def get_realtime_data(self, symbol: str, start_datetime: Optional[str] = None,
                               end_datetime: Optional[str] = None, 
                               data_type: str = "quote") -> pd.DataFrame:
        """
        Get realtime data for a symbol.
        
        Args:
            symbol: Stock symbol
            start_datetime: Start datetime in ISO format (optional)
            end_datetime: End datetime in ISO format (optional)
            data_type: Type of data ('quote' or 'trade')
            
        Returns:
            DataFrame with realtime data
        """
        result = await self._make_request("get_realtime_data", {
            "ticker": symbol,
            "start_datetime": start_datetime,
            "end_datetime": end_datetime,
            "data_type": data_type
        })
        
        if not result.get("data"):
            logger.warning(f"No realtime {data_type} data found for {symbol}")
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(result["data"])
        
        if df.empty:
            return df
        
        # Convert timestamp column to datetime and set as index
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
        
        # Ensure numeric columns are properly typed
        numeric_cols = ['price', 'size', 'bid_price', 'ask_price', 'bid_size', 'ask_size']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Sort by timestamp
        df.sort_index(inplace=True)
        
        # Remove any rows with NaN values
        df.dropna(inplace=True)
        
        logger.info(f"Retrieved {len(df)} realtime {data_type} records for {symbol}")
        return df
    
    async def get_latest_price(self, symbol: str) -> Optional[float]:
        """
        Get the latest price for a symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Latest price or None if not found
        """
        result = await self._make_request("get_latest_price", {
            "ticker": symbol
        })
        
        return result.get("latest_price")
    
    async def health_check(self) -> bool:
        """
        Check if the database server is healthy.
        
        Returns:
            True if server is healthy, False otherwise
        """
        try:
            async with self.session.get(f"{self.base_url}/health") as response:
                return response.status == 200
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
