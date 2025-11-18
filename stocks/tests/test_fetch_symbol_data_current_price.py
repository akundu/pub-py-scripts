#!/usr/bin/env python3
"""
Test suite for current price functionality in fetch_symbol_data.py.
"""

import asyncio
import os
import sys
import pytest
import pandas as pd
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, patch, AsyncMock

# Add the parent directory to the path to import the modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fetch_symbol_data import (
    get_current_price,
    _get_current_price_polygon,
    _get_current_price_alpaca,
    _get_latest_price_with_timestamp,
    get_stock_price_simple
)


class TestCurrentPriceFunctionality:
    """Test current price fetching functionality."""
    
    @pytest.fixture
    def mock_questdb_instance(self):
        """Create a mock QuestDB instance."""
        db_instance = Mock()
        db_instance.get_realtime_data = AsyncMock()
        db_instance.get_stock_data = AsyncMock()
        db_instance.save_realtime_data = AsyncMock()
        return db_instance
    
    @pytest.mark.asyncio
    async def test_get_current_price_from_database_fresh(self, mock_questdb_instance):
        """Test getting current price from database when data is fresh."""
        # Mock fresh realtime data
        fresh_timestamp = datetime.now(timezone.utc) - timedelta(seconds=30)
        mock_realtime_data = pd.DataFrame({
            'price': [150.0],
            'bid_price': [149.5],
            'ask_price': [150.5]
        }, index=[fresh_timestamp])
        mock_realtime_data.index.name = 'timestamp'
        
        mock_questdb_instance.get_realtime_data.return_value = mock_realtime_data
        
        result = await get_current_price(
            symbol="AAPL",
            stock_db_instance=mock_questdb_instance,
            max_age_seconds=600
        )
        
        assert result['symbol'] == "AAPL"
        assert result['price'] == 150.0
        assert result['bid_price'] == 149.5
        assert result['ask_price'] == 150.5
        assert result['source'] == 'database'
    
    @pytest.mark.asyncio
    async def test_get_current_price_from_database_stale(self, mock_questdb_instance):
        """Test getting current price when database data is stale."""
        # Mock stale realtime data
        stale_timestamp = datetime.now(timezone.utc) - timedelta(hours=2)
        mock_realtime_data = pd.DataFrame({
            'price': [150.0],
            'bid_price': [149.5],
            'ask_price': [150.5]
        }, index=[stale_timestamp])
        mock_realtime_data.index.name = 'timestamp'
        
        mock_questdb_instance.get_realtime_data.return_value = mock_realtime_data
        
        with patch('fetch_symbol_data._get_current_price_polygon') as mock_polygon:
            mock_polygon.return_value = {
                'symbol': 'AAPL',
                'price': 155.0,
                'source': 'polygon_quote'
            }
            
            result = await get_current_price(
                symbol="AAPL",
                stock_db_instance=mock_questdb_instance,
                max_age_seconds=600
            )
            
            # Should fetch from API since database data is stale
            assert result['price'] == 155.0
            assert result['source'] == 'polygon_quote'
            mock_polygon.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_current_price_fallback_to_hourly(self, mock_questdb_instance):
        """Test fallback to hourly data when no realtime data."""
        # Mock no realtime data, but hourly data available
        mock_questdb_instance.get_realtime_data.return_value = pd.DataFrame()
        
        hourly_timestamp = datetime.now(timezone.utc) - timedelta(hours=1)
        mock_hourly_data = pd.DataFrame({
            'close': [152.0]
        }, index=[hourly_timestamp])
        mock_hourly_data.index.name = 'datetime'
        
        mock_questdb_instance.get_stock_data.return_value = mock_hourly_data
        
        result = await get_current_price(
            symbol="AAPL",
            stock_db_instance=mock_questdb_instance,
            max_age_seconds=600
        )
        
        assert result['symbol'] == "AAPL"
        assert result['price'] == 152.0
        assert result['source'] == 'database'
    
    @pytest.mark.asyncio
    async def test_get_current_price_fallback_to_daily(self, mock_questdb_instance):
        """Test fallback to daily data when no realtime or hourly data."""
        # Mock no realtime or hourly data, but daily data available
        mock_questdb_instance.get_realtime_data.return_value = pd.DataFrame()
        mock_questdb_instance.get_stock_data.side_effect = [
            pd.DataFrame(),  # No hourly data
            pd.DataFrame({   # Daily data
                'close': [150.0]
            }, index=[datetime.now(timezone.utc).date()])
        ]
        
        result = await get_current_price(
            symbol="AAPL",
            stock_db_instance=mock_questdb_instance,
            max_age_seconds=600
        )
        
        assert result['symbol'] == "AAPL"
        assert result['price'] == 150.0
        assert result['source'] == 'database'
    
    @pytest.mark.asyncio
    async def test_get_current_price_no_database_data(self, mock_questdb_instance):
        """Test fetching from API when no database data."""
        # Mock no data in database
        mock_questdb_instance.get_realtime_data.return_value = pd.DataFrame()
        mock_questdb_instance.get_stock_data.return_value = pd.DataFrame()
        
        with patch('fetch_symbol_data._get_current_price_polygon') as mock_polygon:
            mock_polygon.return_value = {
                'symbol': 'AAPL',
                'price': 155.0,
                'bid_price': 154.5,
                'ask_price': 155.5,
                'source': 'polygon_quote'
            }
            
            result = await get_current_price(
                symbol="AAPL",
                data_source="polygon",
                stock_db_instance=mock_questdb_instance,
                max_age_seconds=600
            )
            
            assert result['price'] == 155.0
            assert result['source'] == 'polygon_quote'
            mock_polygon.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_current_price_polygon_quote(self):
        """Test Polygon quote fetching."""
        mock_quote = Mock()
        mock_quote.bid_price = 150.0
        mock_quote.ask_price = 150.5
        mock_quote.bid_size = 100
        mock_quote.ask_size = 200
        mock_quote.sip_timestamp = 1704067200000000000  # Nanoseconds
        
        with patch('fetch_symbol_data.PolygonRESTClient') as mock_client_class:
            mock_client = Mock()
            mock_client.get_last_quote.return_value = mock_quote
            mock_client_class.return_value = mock_client
            
            result = await _get_current_price_polygon("AAPL")
            
            assert result['symbol'] == "AAPL"
            assert result['price'] == 150.0
            assert result['bid_price'] == 150.0
            assert result['ask_price'] == 150.5
            assert result['source'] == 'polygon_quote'
    
    @pytest.mark.asyncio
    async def test_get_current_price_polygon_trade(self):
        """Test Polygon trade fetching when no quote available."""
        mock_trade = Mock()
        mock_trade.price = 150.25
        mock_trade.size = 1000
        mock_trade.sip_timestamp = 1704067200000000000
        
        with patch('fetch_symbol_data.PolygonRESTClient') as mock_client_class:
            mock_client = Mock()
            mock_client.get_last_quote.return_value = None
            mock_client.get_last_trade.return_value = mock_trade
            mock_client_class.return_value = mock_client
            
            result = await _get_current_price_polygon("AAPL")
            
            assert result['symbol'] == "AAPL"
            assert result['price'] == 150.25
            assert result['source'] == 'polygon_trade'
    
    @pytest.mark.asyncio
    async def test_get_current_price_polygon_daily_fallback(self):
        """Test Polygon daily bar fallback when no quote or trade."""
        mock_bar = Mock()
        mock_bar.close = 150.0
        mock_bar.open = 149.0
        mock_bar.high = 151.0
        mock_bar.low = 148.0
        mock_bar.volume = 1000000
        mock_bar.timestamp = 1704067200000  # Milliseconds
        
        with patch('fetch_symbol_data.PolygonRESTClient') as mock_client_class:
            mock_client = Mock()
            mock_client.get_last_quote.return_value = None
            mock_client.get_last_trade.return_value = None
            mock_client.get_aggs.return_value = [mock_bar]
            mock_client_class.return_value = mock_client
            
            result = await _get_current_price_polygon("AAPL")
            
            assert result['symbol'] == "AAPL"
            assert result['price'] == 150.0
            assert result['source'] == 'polygon_daily'
    
    @pytest.mark.asyncio
    async def test_get_current_price_alpaca_quote(self):
        """Test Alpaca quote fetching."""
        mock_quote_data = {
            'quote': {
                'bp': 150.0,  # bid price
                'ap': 150.5,  # ask price
                'bs': 100,    # bid size
                'as': 200,    # ask size
                't': '2024-01-01T10:00:00Z'
            }
        }
        
        with patch('fetch_symbol_data.aiohttp.ClientSession') as mock_session:
            mock_response = Mock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value=mock_quote_data)
            
            mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.return_value = mock_response
            
            result = await _get_current_price_alpaca("AAPL")
            
            assert result['symbol'] == "AAPL"
            assert result['price'] == 150.0
            assert result['bid_price'] == 150.0
            assert result['ask_price'] == 150.5
            assert result['source'] == 'alpaca_quote'
    
    @pytest.mark.asyncio
    async def test_get_current_price_alpaca_trade(self):
        """Test Alpaca trade fetching when no quote available."""
        mock_trade_data = {
            'trade': {
                'p': 150.25,  # price
                's': 1000,    # size
                't': '2024-01-01T10:00:00Z'
            }
        }
        
        with patch('fetch_symbol_data.aiohttp.ClientSession') as mock_session:
            # Mock quote endpoint returning 404, trade endpoint returning data
            mock_quote_response = Mock()
            mock_quote_response.status = 404
            
            mock_trade_response = Mock()
            mock_trade_response.status = 200
            mock_trade_response.json = AsyncMock(return_value=mock_trade_data)
            
            mock_session.return_value.__aenter__.return_value.get.side_effect = [
                mock_quote_response,  # Quote endpoint
                mock_trade_response   # Trade endpoint
            ]
            
            result = await _get_current_price_alpaca("AAPL")
            
            assert result['symbol'] == "AAPL"
            assert result['price'] == 150.25
            assert result['source'] == 'alpaca_trade'
    
    @pytest.mark.asyncio
    async def test_get_latest_price_with_timestamp_realtime(self, mock_questdb_instance):
        """Test getting latest price with timestamp from realtime data."""
        timestamp = datetime.now(timezone.utc)
        write_timestamp = timestamp - timedelta(seconds=10)
        
        mock_realtime_data = pd.DataFrame({
            'price': [150.0],
            'write_timestamp': [write_timestamp]
        }, index=[timestamp])
        mock_realtime_data.index.name = 'timestamp'
        
        mock_questdb_instance.get_realtime_data.return_value = mock_realtime_data
        
        result = await _get_latest_price_with_timestamp(mock_questdb_instance, "AAPL")
        
        assert result['price'] == 150.0
        assert result['timestamp'] == timestamp
        assert result['write_timestamp'] == write_timestamp
    
    @pytest.mark.asyncio
    async def test_get_latest_price_with_timestamp_hourly(self, mock_questdb_instance):
        """Test getting latest price with timestamp from hourly data."""
        mock_questdb_instance.get_realtime_data.return_value = pd.DataFrame()
        
        timestamp = datetime.now(timezone.utc) - timedelta(hours=1)
        mock_hourly_data = pd.DataFrame({
            'close': [152.0]
        }, index=[timestamp])
        mock_hourly_data.index.name = 'datetime'
        
        mock_questdb_instance.get_stock_data.return_value = mock_hourly_data
        
        result = await _get_latest_price_with_timestamp(mock_questdb_instance, "AAPL")
        
        assert result['price'] == 152.0
        assert result['timestamp'] == timestamp
        assert result['write_timestamp'] is None
    
    def test_get_stock_price_simple(self):
        """Test simple synchronous price fetching."""
        with patch('fetch_symbol_data.get_current_price') as mock_get_current_price:
            mock_get_current_price.return_value = {
                'symbol': 'AAPL',
                'price': 150.0,
                'source': 'database'
            }
            
            result = get_stock_price_simple("AAPL", "polygon")
            
            assert result == 150.0
            mock_get_current_price.assert_called_once()
    
    def test_get_stock_price_simple_error(self):
        """Test simple price fetching with error."""
        with patch('fetch_symbol_data.get_current_price') as mock_get_current_price:
            mock_get_current_price.side_effect = Exception("API Error")
            
            result = get_stock_price_simple("AAPL", "polygon")
            
            assert result is None
    
    @pytest.mark.asyncio
    async def test_current_price_with_write_timestamp_priority(self, mock_questdb_instance):
        """Test that write_timestamp is used for age calculation when available."""
        # Mock data with both original timestamp and write_timestamp
        original_timestamp = datetime.now(timezone.utc) - timedelta(hours=2)
        write_timestamp = datetime.now(timezone.utc) - timedelta(seconds=30)
        
        mock_realtime_data = pd.DataFrame({
            'price': [150.0],
            'write_timestamp': [write_timestamp]
        }, index=[original_timestamp])
        mock_realtime_data.index.name = 'timestamp'
        
        mock_questdb_instance.get_realtime_data.return_value = mock_realtime_data
        
        result = await get_current_price(
            symbol="AAPL",
            stock_db_instance=mock_questdb_instance,
            max_age_seconds=600
        )
        
        # Should use write_timestamp for age calculation (30 seconds < 600 seconds)
        assert result['price'] == 150.0
        assert result['source'] == 'database'
    
    @pytest.mark.asyncio
    async def test_current_price_timezone_handling(self, mock_questdb_instance):
        """Test timezone handling in current price fetching."""
        # Mock data with timezone-aware timestamp
        timestamp = datetime.now(timezone.utc)
        mock_realtime_data = pd.DataFrame({
            'price': [150.0]
        }, index=[timestamp])
        mock_realtime_data.index.name = 'timestamp'
        
        mock_questdb_instance.get_realtime_data.return_value = mock_realtime_data
        
        result = await get_current_price(
            symbol="AAPL",
            stock_db_instance=mock_questdb_instance,
            max_age_seconds=600
        )
        
        assert result['price'] == 150.0
        # Verify timestamp is properly handled
        assert 'timestamp' in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])





