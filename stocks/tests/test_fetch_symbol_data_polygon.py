#!/usr/bin/env python3
"""
Test suite for Polygon.io data fetching functionality in fetch_symbol_data.py.
"""

import asyncio
import os
import sys
import pytest
import pandas as pd
from datetime import datetime, timezone
from unittest.mock import Mock, patch, AsyncMock

# Add the parent directory to the path to import the modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fetch_symbol_data import (
    fetch_polygon_data,
    _fetch_polygon_chunk,
    _fetch_polygon_paginated,
    _get_polygon_timespan
)


class TestPolygonDataFetching:
    """Test Polygon.io data fetching functionality."""
    
    @pytest.fixture
    def mock_polygon_client(self):
        """Create a mock Polygon client."""
        client = Mock()
        return client
    
    def test_get_polygon_timespan(self):
        """Test timespan conversion for Polygon API."""
        assert _get_polygon_timespan("daily") == "day"
        assert _get_polygon_timespan("hourly") == "hour"
        
        with pytest.raises(ValueError):
            _get_polygon_timespan("invalid")
    
    @pytest.mark.asyncio
    async def test_fetch_polygon_data_daily(self, mock_polygon_client):
        """Test fetching daily data from Polygon."""
        # Mock the client response
        mock_response = [
            Mock(timestamp=1704067200000, open=100.0, high=105.0, low=95.0, close=102.0, volume=1000000),
            Mock(timestamp=1704153600000, open=102.0, high=107.0, low=97.0, close=104.0, volume=1100000)
        ]
        
        with patch('fetch_symbol_data.PolygonRESTClient') as mock_client_class:
            mock_client_class.return_value = mock_polygon_client
            mock_polygon_client.get_aggs.return_value = mock_response
            
            result = await fetch_polygon_data(
                symbol="AAPL",
                timeframe="daily",
                start_date="2024-01-01",
                end_date="2024-01-02",
                api_key="test_key"
            )
            
            assert not result.empty
            assert len(result) == 2
            assert list(result.columns) == ['open', 'high', 'low', 'close', 'volume']
            assert result.index.name == 'date'
    
    @pytest.mark.asyncio
    async def test_fetch_polygon_data_hourly(self, mock_polygon_client):
        """Test fetching hourly data from Polygon."""
        # Mock the client response
        mock_response = [
            Mock(timestamp=1704067200000, open=100.0, high=105.0, low=95.0, close=102.0, volume=100000),
            Mock(timestamp=1704070800000, open=102.0, high=107.0, low=97.0, close=104.0, volume=110000)
        ]
        
        with patch('fetch_symbol_data.PolygonRESTClient') as mock_client_class:
            mock_client_class.return_value = mock_polygon_client
            mock_polygon_client.get_aggs.return_value = mock_response
            
            result = await fetch_polygon_data(
                symbol="AAPL",
                timeframe="hourly",
                start_date="2024-01-01",
                end_date="2024-01-01",
                api_key="test_key"
            )
            
            assert not result.empty
            assert len(result) == 2
            assert list(result.columns) == ['open', 'high', 'low', 'close', 'volume']
            assert result.index.name == 'datetime'
            # Hourly data should have timezone info
            assert result.index.tz is not None
    
    @pytest.mark.asyncio
    async def test_fetch_polygon_data_empty_response(self, mock_polygon_client):
        """Test handling of empty response from Polygon."""
        with patch('fetch_symbol_data.PolygonRESTClient') as mock_client_class:
            mock_client_class.return_value = mock_polygon_client
            mock_polygon_client.get_aggs.return_value = []
            
            result = await fetch_polygon_data(
                symbol="INVALID",
                timeframe="daily",
                start_date="2024-01-01",
                end_date="2024-01-01",
                api_key="test_key"
            )
            
            assert result.empty
    
    @pytest.mark.asyncio
    async def test_fetch_polygon_data_chunked(self, mock_polygon_client):
        """Test chunked data fetching."""
        # Mock responses for different chunks
        chunk1_response = [
            Mock(timestamp=1704067200000, open=100.0, high=105.0, low=95.0, close=102.0, volume=1000000)
        ]
        chunk2_response = [
            Mock(timestamp=1704153600000, open=102.0, high=107.0, low=97.0, close=104.0, volume=1100000)
        ]
        
        with patch('fetch_symbol_data.PolygonRESTClient') as mock_client_class:
            mock_client_class.return_value = mock_polygon_client
            mock_polygon_client.get_aggs.side_effect = [chunk1_response, chunk2_response]
            
            result = await fetch_polygon_data(
                symbol="AAPL",
                timeframe="daily",
                start_date="2024-01-01",
                end_date="2024-01-02",
                api_key="test_key",
                chunk_size="daily"
            )
            
            assert not result.empty
            assert len(result) == 2
            assert mock_polygon_client.get_aggs.call_count == 2
    
    @pytest.mark.asyncio
    async def test_fetch_polygon_data_auto_chunk_size(self, mock_polygon_client):
        """Test automatic chunk size selection."""
        mock_response = [Mock(timestamp=1704067200000, open=100.0, high=105.0, low=95.0, close=102.0, volume=1000000)]
        
        with patch('fetch_symbol_data.PolygonRESTClient') as mock_client_class:
            mock_client_class.return_value = mock_polygon_client
            mock_polygon_client.get_aggs.return_value = mock_response
            
            # Test with auto chunk size for hourly data over long period
            result = await fetch_polygon_data(
                symbol="AAPL",
                timeframe="hourly",
                start_date="2024-01-01",
                end_date="2024-06-01",  # 5 months - should trigger monthly chunks
                api_key="test_key",
                chunk_size="auto"
            )
            
            assert not result.empty
            # Should use monthly chunks for this long period
    
    @pytest.mark.asyncio
    async def test_fetch_polygon_data_error_handling(self, mock_polygon_client):
        """Test error handling in Polygon data fetching."""
        with patch('fetch_symbol_data.PolygonRESTClient') as mock_client_class:
            mock_client_class.return_value = mock_polygon_client
            mock_polygon_client.get_aggs.side_effect = Exception("API Error")
            
            with pytest.raises(Exception, match="API Error"):
                await fetch_polygon_data(
                    symbol="AAPL",
                    timeframe="daily",
                    start_date="2024-01-01",
                    end_date="2024-01-01",
                    api_key="test_key"
                )
    
    @pytest.mark.asyncio
    async def test_fetch_polygon_chunk(self, mock_polygon_client):
        """Test individual chunk fetching."""
        mock_response = [
            Mock(timestamp=1704067200000, open=100.0, high=105.0, low=95.0, close=102.0, volume=1000000)
        ]
        mock_polygon_client.get_aggs.return_value = mock_response
        
        result = await _fetch_polygon_chunk(
            client=mock_polygon_client,
            symbol="AAPL",
            timespan="day",
            start_date="2024-01-01",
            end_date="2024-01-01"
        )
        
        assert result == mock_response
        mock_polygon_client.get_aggs.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_fetch_polygon_chunk_error(self, mock_polygon_client):
        """Test error handling in chunk fetching."""
        mock_polygon_client.get_aggs.side_effect = Exception("Chunk error")
        
        result = await _fetch_polygon_chunk(
            client=mock_polygon_client,
            symbol="AAPL",
            timespan="day",
            start_date="2024-01-01",
            end_date="2024-01-01"
        )
        
        assert result == []
    
    @pytest.mark.asyncio
    async def test_fetch_polygon_paginated(self, mock_polygon_client):
        """Test paginated data fetching."""
        # Mock paginated responses
        page1_response = [
            Mock(timestamp=1704067200000, open=100.0, high=105.0, low=95.0, close=102.0, volume=1000000)
        ]
        page2_response = [
            Mock(timestamp=1704153600000, open=102.0, high=107.0, low=97.0, close=104.0, volume=1100000)
        ]
        page3_response = []  # Empty response to stop pagination
        
        mock_polygon_client.get_aggs.side_effect = [page1_response, page2_response, page3_response]
        
        result = await _fetch_polygon_paginated(
            client=mock_polygon_client,
            symbol="AAPL",
            timespan="day",
            start_date="2024-01-01",
            end_date="2024-01-02"
        )
        
        assert len(result) == 2
        assert mock_polygon_client.get_aggs.call_count == 3  # Called 3 times (2 with data, 1 empty)
    
    @pytest.mark.asyncio
    async def test_fetch_polygon_data_date_validation(self, mock_polygon_client):
        """Test date validation and formatting."""
        mock_response = [Mock(timestamp=1704067200000, open=100.0, high=105.0, low=95.0, close=102.0, volume=1000000)]
        
        with patch('fetch_symbol_data.PolygonRESTClient') as mock_client_class:
            mock_client_class.return_value = mock_polygon_client
            mock_polygon_client.get_aggs.return_value = mock_response
            
            # Test with ISO format dates
            result = await fetch_polygon_data(
                symbol="AAPL",
                timeframe="daily",
                start_date="2024-01-01T00:00:00",
                end_date="2024-01-01T23:59:59",
                api_key="test_key"
            )
            
            assert not result.empty
            # Verify the client was called with properly formatted dates
            call_args = mock_polygon_client.get_aggs.call_args
            assert call_args[1]['from_'] == '2024-01-01'
            assert call_args[1]['to'] == '2024-01-01'
    
    @pytest.mark.asyncio
    async def test_fetch_polygon_data_invalid_dates(self, mock_polygon_client):
        """Test handling of invalid date formats."""
        with patch('fetch_symbol_data.PolygonRESTClient') as mock_client_class:
            mock_client_class.return_value = mock_polygon_client
            
            with pytest.raises(Exception):
                await fetch_polygon_data(
                    symbol="AAPL",
                    timeframe="daily",
                    start_date="invalid-date",
                    end_date="2024-01-01",
                    api_key="test_key"
                )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])





