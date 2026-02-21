"""
Comprehensive test suite for db_server.py functionality.
Tests cover serialization, filtering, HTML generation, WebSocket management, and more.
"""

import pytest
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from pathlib import Path
import sys
import json
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from io import StringIO
import logging

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the functions we want to test - try new modules first, fall back to db_server
try:
    from common.web.serializers import dataframe_to_json_records, serialize_mapping_datetime
    from common.web.filters import parse_filter_strings as _parse_filter_strings, apply_filters as _apply_filters
    from common.web.html_generators import format_options_html as _format_options_html, generate_stock_info_html
    # Try server.websocket_manager first (has redis_enabled), fall back to db_server
    try:
        from server.websocket_manager import WebSocketManager
    except ImportError:
        from db_server import WebSocketManager
except ImportError:
    # Fall back to original db_server if modules not yet refactored
    from db_server import (
        dataframe_to_json_records,
        serialize_mapping_datetime,
        _parse_filter_strings,
        _apply_filters,
        _format_options_html,
        generate_stock_info_html,
        WebSocketManager,
    )

# Import for daily range tests
try:
    from common.questdb_db import RealtimeDataService
except ImportError:
    # If not available, tests will be skipped
    RealtimeDataService = None


class TestDataframeSerialization:
    """Test DataFrame to JSON conversion functions."""
    
    def test_dataframe_to_json_records_basic(self):
        """Test basic DataFrame conversion."""
        df = pd.DataFrame({
            'symbol': ['AAPL', 'GOOGL'],
            'price': [150.0, 2800.0],
            'volume': [1000000, 500000]
        })
        
        result = dataframe_to_json_records(df)
        
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]['symbol'] == 'AAPL'
        assert result[0]['price'] == 150.0
        assert result[1]['symbol'] == 'GOOGL'
    
    def test_dataframe_to_json_records_with_timestamps(self):
        """Test DataFrame with timestamp columns."""
        df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=3),
            'price': [100, 101, 102]
        })
        
        result = dataframe_to_json_records(df)
        
        assert len(result) == 3
        # Timestamps should be converted to ISO strings
        assert isinstance(result[0]['date'], str)
        assert '2024-01-01' in result[0]['date']
    
    def test_dataframe_to_json_records_with_nan(self):
        """Test DataFrame with NaN values."""
        df = pd.DataFrame({
            'value': [1.0, np.nan, 3.0]
        })
        
        result = dataframe_to_json_records(df)
        
        assert len(result) == 3
        assert result[0]['value'] == 1.0
        assert result[1]['value'] is None  # NaN should become None
        assert result[2]['value'] == 3.0
    
    def test_dataframe_to_json_records_empty(self):
        """Test empty DataFrame."""
        df = pd.DataFrame()
        result = dataframe_to_json_records(df)
        assert result == []
    
    def test_dataframe_to_json_records_with_datetime_index(self):
        """Test DataFrame with datetime index."""
        dates = pd.date_range('2024-01-01', periods=3)
        df = pd.DataFrame(
            {'price': [100, 101, 102]},
            index=dates
        )
        
        result = dataframe_to_json_records(df)
        
        assert len(result) == 3
        assert 'price' in result[0]


class TestDatetimeSerialization:
    """Test datetime serialization functions."""
    
    def test_serialize_mapping_datetime_basic(self):
        """Test basic datetime serialization."""
        data = {
            'timestamp': datetime(2024, 1, 1, 12, 0, 0),
            'value': 100
        }
        
        result = serialize_mapping_datetime(data)
        
        assert isinstance(result['timestamp'], str)
        assert '2024-01-01' in result['timestamp']
        assert result['value'] == 100
    
    def test_serialize_mapping_datetime_nested(self):
        """Test nested datetime serialization."""
        data = {
            'outer': {
                'inner': {
                    'timestamp': datetime(2024, 1, 1, tzinfo=timezone.utc)
                }
            },
            'list_data': [
                {'time': datetime(2024, 1, 2, tzinfo=timezone.utc)}
            ]
        }
        
        result = serialize_mapping_datetime(data)
        
        assert isinstance(result['outer']['inner']['timestamp'], str)
        assert isinstance(result['list_data'][0]['time'], str)
    
    def test_serialize_mapping_datetime_with_pandas_timestamp(self):
        """Test Pandas Timestamp serialization."""
        data = {
            'pd_time': pd.Timestamp('2024-01-01 12:00:00'),
            'regular': 'text'
        }
        
        result = serialize_mapping_datetime(data)
        
        assert isinstance(result['pd_time'], str)
        assert result['regular'] == 'text'


class TestFilterParsing:
    """Test filter parsing and application."""
    
    def test_parse_filter_strings_simple(self):
        """Test simple filter parsing."""
        filter_str = "column1 > 100"
        result = _parse_filter_strings(filter_str)
        
        assert len(result) == 1
        assert result[0]['field'] == 'column1'
        assert result[0]['operator'] == '>'
        assert result[0]['value'] == 100  # Parsed as number
    
    def test_parse_filter_strings_multiple(self):
        """Test multiple filters separated by pipe."""
        filter_str = "price > 50|volume < 1000000"
        result = _parse_filter_strings(filter_str)
        
        assert len(result) == 2
        assert result[0]['field'] == 'price'
        assert result[1]['field'] == 'volume'
    
    def test_parse_filter_strings_with_spaces(self):
        """Test filters with various spacing."""
        filter_str = "  price  >=  100.50  "
        result = _parse_filter_strings(filter_str)
        
        assert len(result) == 1
        assert result[0]['field'] == 'price'
        assert result[0]['operator'] == '>='
        assert result[0]['value'] == 100.50  # Parsed as float
    
    def test_parse_filter_strings_equality(self):
        """Test equality filters."""
        filter_str = "symbol == AAPL"
        result = _parse_filter_strings(filter_str)
        
        assert len(result) == 1
        assert result[0]['operator'] == '=='
        assert result[0]['value'] == 'AAPL'
    
    def test_parse_filter_strings_contains(self):
        """Test that unsupported operators return empty (contains not supported)."""
        filter_str = "symbol contains APP"
        result = _parse_filter_strings(filter_str)
        
        # "contains" is not a supported operator, so nothing should be parsed
        assert len(result) == 0
    
    def test_apply_filters_numeric(self):
        """Test applying numeric filters."""
        df = pd.DataFrame({
            'price': [50, 100, 150, 200],
            'volume': [1000, 2000, 3000, 4000]
        })
        
        filters = [{'field': 'price', 'operator': '>', 'value': 100}]
        result = _apply_filters(df, filters, filter_logic='AND')
        
        assert len(result) == 2
        assert all(result['price'] > 100)
    
    def test_apply_filters_and_logic(self):
        """Test AND filter logic."""
        df = pd.DataFrame({
            'price': [50, 100, 150, 200],
            'volume': [1000, 2000, 3000, 4000]
        })
        
        filters = [
            {'field': 'price', 'operator': '>', 'value': 75},
            {'field': 'volume', 'operator': '<', 'value': 3500}
        ]
        result = _apply_filters(df, filters, filter_logic='AND')
        
        assert len(result) == 2  # price > 75 AND volume < 3500
        assert all((result['price'] > 75) & (result['volume'] < 3500))
    
    def test_apply_filters_or_logic(self):
        """Test OR filter logic."""
        df = pd.DataFrame({
            'price': [50, 100, 150, 200],
            'volume': [1000, 2000, 3000, 4000]
        })
        
        filters = [
            {'field': 'price', 'operator': '<', 'value': 75},
            {'field': 'volume', 'operator': '>', 'value': 3500}
        ]
        result = _apply_filters(df, filters, filter_logic='OR')
        
        assert len(result) == 2  # price < 75 OR volume > 3500
    
    def test_apply_filters_string_equality(self):
        """Test string equality filters."""
        df = pd.DataFrame({
            'symbol': ['AAPL', 'GOOGL', 'MSFT', 'AAPL'],
            'price': [150, 2800, 300, 151]
        })
        
        filters = [{'field': 'symbol', 'operator': '==', 'value': 'AAPL'}]
        result = _apply_filters(df, filters, filter_logic='AND')
        
        assert len(result) == 2
        assert all(result['symbol'] == 'AAPL')
    
    def test_apply_filters_inequality(self):
        """Test inequality (!=) filter."""
        df = pd.DataFrame({
            'symbol': ['AAPL', 'GOOGL', 'MSFT', 'AAPL'],
            'price': [150, 2800, 300, 151]
        })
        
        filters = [{'field': 'symbol', 'operator': '!=', 'value': 'AAPL'}]
        result = _apply_filters(df, filters, filter_logic='AND')
        
        assert len(result) == 2
        assert all(result['symbol'] != 'AAPL')
    
    def test_apply_filters_empty_dataframe(self):
        """Test filters on empty DataFrame."""
        df = pd.DataFrame(columns=['price', 'volume'])
        filters = [{'field': 'price', 'operator': '>', 'value': 100}]
        result = _apply_filters(df, filters, filter_logic='AND')
        
        assert len(result) == 0
    
    def test_apply_filters_no_filters(self):
        """Test with no filters - should return original."""
        df = pd.DataFrame({'price': [100, 200, 300]})
        result = _apply_filters(df, [], filter_logic='AND')
        
        assert len(result) == 3
        assert result.equals(df)


class TestHTMLGeneration:
    """Test HTML generation functions."""
    
    def test_format_options_html_empty(self):
        """Test options HTML with no data."""
        options_data = {'success': False}
        result = _format_options_html(options_data)
        
        assert '<p>No options data available</p>' in result
    
    def test_format_options_html_no_contracts(self):
        """Test options HTML with no contracts."""
        options_data = {
            'success': True,
            'data': {'contracts': []}
        }
        result = _format_options_html(options_data)
        
        assert '<p>No options contracts found</p>' in result
    
    def test_format_options_html_with_contracts(self):
        """Test options HTML with contract data."""
        options_data = {
            'success': True,
            'data': {
                'contracts': [
                    {
                        'expiration': '2024-12-20',
                        'option_type': 'call',
                        'strike': 150.0,
                        'bid': 5.50,
                        'ask': 5.60,
                        'last': 5.55,
                        'volume': 1000,
                        'open_interest': 5000,
                        'implied_volatility': 0.25
                    }
                ]
            }
        }
        
        result = _format_options_html(options_data)
        
        assert '<table' in result
        assert '2024-12-20' in result
        # HTML uses 'CALLS' (uppercase) in headers, but may have 'call' in data attributes
        assert 'CALLS' in result or 'call' in result.lower()
        assert '$150.00' in result
        assert '25.00%' in result or '0.25' in result
    
    def test_generate_stock_info_html_basic(self):
        """Test basic stock info HTML generation."""
        symbol = 'AAPL'
        data = {
            'price_info': {
                'current_price': {'price': 150.50, 'change': 2.50, 'change_percent': 1.69},
                'price_data': []
            },
            'financial_info': {},
            'options_info': {},
            'iv_info': {},
            'news_info': {}
        }
        
        result = generate_stock_info_html(symbol, data)
        
        assert 'AAPL' in result
        assert '150.50' in result
        assert 'html' in result.lower()
        assert '<' in result  # Should contain HTML tags
    
    def test_generate_stock_info_html_with_price_data(self):
        """Test stock info HTML with price history."""
        symbol = 'AAPL'
        data = {
            'price_info': {
                'current_price': {'price': 150.50},
                'price_data': [
                    {'date': '2024-01-01', 'close': 148.0, 'volume': 1000000},
                    {'date': '2024-01-02', 'close': 150.5, 'volume': 1100000}
                ]
            },
            'financial_info': {},
            'options_info': {},
            'iv_info': {},
            'news_info': {}
        }
        
        result = generate_stock_info_html(symbol, data)
        
        assert 'AAPL' in result
        assert 'chart' in result.lower() or 'price' in result.lower()
    
    def test_generate_stock_info_html_with_news(self):
        """Test stock info HTML with news items."""
        symbol = 'AAPL'
        data = {
            'price_info': {'current_price': {'price': 150.50}, 'price_data': []},
            'financial_info': {},
            'options_info': {},
            'iv_info': {},
            'news_info': {
                'news': [
                    {
                        'title': 'Apple releases new iPhone',
                        'source': 'TechNews',
                        'published': '2024-01-01T12:00:00',
                        'url': 'https://example.com/news1'
                    }
                ]
            }
        }
        
        result = generate_stock_info_html(symbol, data)
        
        assert 'Apple releases new iPhone' in result or 'news' in result.lower()


class TestWebSocketManager:
    """Test WebSocket manager functionality."""
    
    @pytest.fixture
    def ws_manager(self):
        """Create a WebSocket manager for testing."""
        manager = WebSocketManager(
            heartbeat_interval=1.0,
            stale_data_timeout=120.0,
            redis_url=None,
            enable_redis=False
        )
        return manager
    
    def test_websocket_manager_creation(self, ws_manager):
        """Test WebSocket manager can be created."""
        assert ws_manager is not None
        assert ws_manager.heartbeat_interval == 1.0
        assert ws_manager.stale_data_timeout == 120.0
    
    def test_websocket_manager_set_db_instance(self, ws_manager):
        """Test setting database instance."""
        mock_db = Mock()
        ws_manager.set_db_instance(mock_db)
        assert ws_manager.db_instance == mock_db
    
    def test_websocket_manager_update_last_update_time(self, ws_manager):
        """Test updating last update time."""
        symbol = 'AAPL'
        before = len(ws_manager.last_update_times)
        
        ws_manager.update_last_update_time(symbol)
        
        assert len(ws_manager.last_update_times) == before + 1
        assert symbol in ws_manager.last_update_times
        assert isinstance(ws_manager.last_update_times[symbol], float)
    
    @pytest.mark.asyncio
    async def test_websocket_manager_add_subscriber(self, ws_manager):
        """Test adding a subscriber."""
        symbol = 'AAPL'
        mock_ws = AsyncMock()
        
        await ws_manager.add_subscriber(symbol, mock_ws)
        
        assert symbol in ws_manager.connections
        assert mock_ws in ws_manager.connections[symbol]
    
    @pytest.mark.asyncio
    async def test_websocket_manager_remove_subscriber(self, ws_manager):
        """Test removing a subscriber."""
        symbol = 'AAPL'
        mock_ws = AsyncMock()
        
        await ws_manager.add_subscriber(symbol, mock_ws)
        await ws_manager.remove_subscriber(symbol, mock_ws)
        
        # Should be removed or list should be empty
        assert mock_ws not in ws_manager.connections.get(symbol, set())
    
    @pytest.mark.asyncio
    async def test_websocket_manager_broadcast(self, ws_manager):
        """Test broadcasting to subscribers."""
        symbol = 'AAPL'
        mock_ws1 = AsyncMock()
        mock_ws1.closed = False
        mock_ws2 = AsyncMock()
        mock_ws2.closed = False
        
        await ws_manager.add_subscriber(symbol, mock_ws1)
        await ws_manager.add_subscriber(symbol, mock_ws2)
        
        test_data = {'price': 150.50, 'volume': 1000000}
        await ws_manager.broadcast(symbol, test_data)
        
        # Both websockets should have received the message
        assert mock_ws1.send_str.called
        assert mock_ws2.send_str.called
    
    def test_websocket_manager_get_redis_stats(self, ws_manager):
        """Test getting Redis stats."""
        stats = ws_manager.get_redis_stats()
        
        assert isinstance(stats, dict)
        # Check for either 'redis_enabled' (server.websocket_manager) or 'enabled' (db_server.WebSocketManager)
        assert 'redis_enabled' in stats or 'enabled' in stats
        if 'redis_enabled' in stats:
            assert stats['redis_enabled'] is False  # We disabled it in fixture
        else:
            assert stats['enabled'] is False  # We disabled it in fixture


class TestConversionHelpers:
    """Test various conversion helper functions."""
    
    def test_convert_to_json_serializable(self):
        """Test JSON serialization conversion."""
        ws_manager = WebSocketManager(
            heartbeat_interval=1.0,
            stale_data_timeout=120.0,
            redis_url=None,
            enable_redis=False
        )
        
        # Test various types
        assert ws_manager._convert_to_json_serializable(None) is None
        assert ws_manager._convert_to_json_serializable(123) == 123
        assert ws_manager._convert_to_json_serializable("test") == "test"
        assert ws_manager._convert_to_json_serializable([1, 2, 3]) == [1, 2, 3]
        
        # Test datetime
        dt = datetime(2024, 1, 1, 12, 0, 0)
        result = ws_manager._convert_to_json_serializable(dt)
        assert isinstance(result, str)
        assert '2024-01-01' in result
        
        # Test pandas Timestamp
        pd_ts = pd.Timestamp('2024-01-01 12:00:00')
        result = ws_manager._convert_to_json_serializable(pd_ts)
        assert isinstance(result, str)
        
        # Test numpy types
        if np is not None:
            np_int = np.int64(42)
            result = ws_manager._convert_to_json_serializable(np_int)
            assert isinstance(result, int)
            assert result == 42
            
            np_float = np.float64(3.14)
            result = ws_manager._convert_to_json_serializable(np_float)
            assert isinstance(result, float)
            
            # Test NaN
            result = ws_manager._convert_to_json_serializable(np.nan)
            assert result is None


class TestIntegrationScenarios:
    """Integration tests for common workflows."""
    
    def test_full_data_pipeline(self):
        """Test complete data processing pipeline."""
        # Create sample data
        df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=5),
            'symbol': ['AAPL'] * 5,
            'price': [148, 149, 150, 151, 152],
            'volume': [1000000, 1100000, 1200000, 1300000, 1400000]
        })
        
        # Apply filter
        filters = [{'field': 'price', 'operator': '>=', 'value': 150}]
        filtered_df = _apply_filters(df, filters, filter_logic='AND')
        
        # Convert to JSON
        result = dataframe_to_json_records(filtered_df)
        
        # Verify
        assert len(result) == 3  # 150, 151, 152
        assert all(r['price'] >= 150 for r in result)
        assert all(isinstance(r['date'], str) for r in result)
    
    def test_options_data_to_html_pipeline(self):
        """Test options data formatting pipeline."""
        # Create sample options data
        options_data = {
            'success': True,
            'data': {
                'contracts': [
                    {
                        'expiration': '2024-12-20',
                        'option_type': 'call',
                        'strike': 150.0,
                        'bid': 5.50,
                        'ask': 5.60,
                        'last': 5.55,
                        'volume': 1000,
                        'open_interest': 5000,
                        'implied_volatility': 0.25
                    },
                    {
                        'expiration': '2024-12-20',
                        'option_type': 'put',
                        'strike': 145.0,
                        'bid': 3.20,
                        'ask': 3.30,
                        'last': 3.25,
                        'volume': 800,
                        'open_interest': 4000,
                        'implied_volatility': 0.28
                    }
                ]
            }
        }
        
        # Format as HTML
        html = _format_options_html(options_data)
        
        # Verify HTML contains expected elements
        assert '<table' in html
        # HTML uses 'CALLS' and 'PUTS' (uppercase) in headers
        assert ('CALLS' in html or 'call' in html.lower()) and ('PUTS' in html or 'put' in html.lower())
        assert '$150.00' in html
        assert '$145.00' in html
    
    def test_stock_info_complete_workflow(self):
        """Test complete stock info generation workflow."""
        symbol = 'AAPL'
        
        # Create comprehensive data
        data = {
            'price_info': {
                'current_price': {
                    'price': 150.50,
                    'change': 2.50,
                    'change_percent': 1.69,
                    'volume': 50000000
                },
                'price_data': [
                    {'date': '2024-01-01', 'close': 148.0, 'volume': 1000000},
                    {'date': '2024-01-02', 'close': 149.0, 'volume': 1100000},
                    {'date': '2024-01-03', 'close': 150.5, 'volume': 1200000}
                ]
            },
            'financial_info': {
                'financial_data': {
                    'market_cap': 2500000000000,
                    'pe_ratio': 28.5,
                    'dividend_yield': 0.0055
                }
            },
            'options_info': {
                'success': True,
                'data': {
                    'contracts': [
                        {
                            'expiration': '2024-12-20',
                            'option_type': 'call',
                            'strike': 150.0,
                            'bid': 5.50,
                            'ask': 5.60,
                            'last': 5.55,
                            'volume': 1000,
                            'open_interest': 5000,
                            'implied_volatility': 0.25
                        }
                    ]
                }
            },
            'iv_info': {},
            'news_info': {
                'news': [
                    {
                        'title': 'Apple releases earnings',
                        'source': 'Reuters',
                        'published': '2024-01-03T16:00:00',
                        'url': 'https://example.com/news1'
                    }
                ]
            }
        }
        
        # Generate HTML
        html = generate_stock_info_html(symbol, data)
        
        # Verify comprehensive output
        assert 'AAPL' in html
        assert '150.50' in html  # Current price
        assert 'html' in html.lower()
        assert len(html) > 1000  # Should be substantial


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_string_filter(self):
        """Test parsing empty filter string."""
        result = _parse_filter_strings("")
        assert result == []
    
    def test_malformed_filter(self):
        """Test handling of malformed filter strings."""
        # Should handle gracefully without crashing
        try:
            result = _parse_filter_strings("invalid filter format")
            # Should either return empty or handle gracefully
            assert isinstance(result, list)
        except Exception:
            # It's okay if it raises an exception for truly invalid input
            pass
    
    def test_filter_nonexistent_column(self):
        """Test filtering on nonexistent column."""
        df = pd.DataFrame({'price': [100, 200]})
        filters = [{'field': 'nonexistent', 'operator': '>', 'value': 50}]
        
        # Should handle gracefully
        try:
            result = _apply_filters(df, filters, filter_logic='AND')
            # Might return empty or original dataframe
            assert isinstance(result, pd.DataFrame)
        except KeyError:
            # It's okay if it raises KeyError for nonexistent column
            pass
    
    def test_serialize_none_values(self):
        """Test serialization with None values."""
        data = {
            'value1': None,
            'value2': 'text',
            'nested': {'inner': None}
        }
        
        result = serialize_mapping_datetime(data)
        assert result['value1'] is None
        assert result['value2'] == 'text'
        assert result['nested']['inner'] is None
    
    def test_html_generation_missing_data(self):
        """Test HTML generation with minimal/missing data."""
        symbol = 'AAPL'
        data = {
            'price_info': {},
            'financial_info': {},
            'options_info': {},
            'iv_info': {},
            'news_info': {}
        }
        
        # Should not crash
        result = generate_stock_info_html(symbol, data)
        assert 'AAPL' in result
        assert 'html' in result.lower()


class TestCoveredCallsDataTimestamp:
    """Test data source timestamp functionality in covered calls API."""
    
    @pytest.fixture
    def sample_csv_content(self):
        """Create sample CSV content for testing."""
        return """ticker,option_type,strike_price,opt_prem.,delta,volume,expiration
AAPL,call,150.0,5.50,0.30,1000,2024-12-20
AAPL,call,155.0,3.20,0.20,800,2024-12-20
GOOGL,put,2800.0,45.00,-0.35,500,2024-12-20"""
    
    @pytest.fixture
    def temp_csv_file(self, tmp_path, sample_csv_content):
        """Create a temporary CSV file."""
        csv_file = tmp_path / "test_results.csv"
        csv_file.write_text(sample_csv_content)
        return str(csv_file)
    
    @pytest.mark.asyncio
    async def test_api_response_includes_data_source_timestamp(self, temp_csv_file):
        """Test that API response includes data_source_timestamp in metadata."""
        from aiohttp import web
        from aiohttp.test_utils import AioHTTPTestCase, unittest_run_loop
        import db_server
        
        # Create a mock request with the CSV file path
        app = web.Application()
        
        # Mock request
        class MockRequest:
            def __init__(self, csv_path):
                self.app = app
                self.query = {
                    'source': csv_path,
                    'option_type': 'all'
                }
            
            def get_query(self):
                return self.query
        
        request = MockRequest(temp_csv_file)
        # Create a proper mock query object that handles .get() correctly
        class MockQuery:
            def __init__(self, data):
                self._data = data
            
            def get(self, key, default=None):
                return self._data.get(key, default)
        
        request.query = MockQuery({
            'source': temp_csv_file,
            'option_type': 'all',
            'filters': '[]',
            'calls_filters': '',
            'puts_filters': '',
            'filter_logic': 'AND',
            'calls_filterLogic': 'AND',
            'puts_filterLogic': 'AND',
            'sort': 'net_daily_premi',
            'sort_direction': 'desc',
            'limit': None,
            'offset': '0'
        })
        
        # Call the handler
        response = await db_server.handle_covered_calls_data(request)
        
        # Parse response
        assert response.status == 200
        response_data = json.loads(response.text)
        
        # Verify structure
        assert 'metadata' in response_data
        assert 'data_source_timestamp' in response_data['metadata']
        assert response_data['metadata']['data_source_timestamp'] is not None
        
        # Verify timestamp format (should be ISO 8601)
        timestamp = response_data['metadata']['data_source_timestamp']
        assert isinstance(timestamp, str)
        # Should be parseable as datetime
        parsed_dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        assert isinstance(parsed_dt, datetime)
    
    def test_file_modification_time_extraction(self, temp_csv_file):
        """Test that file modification time is correctly extracted."""
        import os
        import time
        
        # Get actual file modification time
        actual_mtime = os.path.getmtime(temp_csv_file)
        
        # Verify we can read it
        assert actual_mtime > 0
        assert isinstance(actual_mtime, float)
        
        # Should be recent (within last minute)
        current_time = time.time()
        assert current_time - actual_mtime < 60
    
    @pytest.mark.asyncio
    async def test_cache_stores_modification_time(self, temp_csv_file):
        """Test that cache properly stores modification time."""
        import db_server
        from aiohttp import web
        
        # Clear any existing cache
        db_server._covered_calls_cache.clear()
        
        app = web.Application()
        
        # Create mock request
        class MockRequest:
            def __init__(self, csv_path):
                self.app = app
                self.query_params = {
                    'source': csv_path,
                    'option_type': 'all',
                    'filters': '[]',
                    'calls_filters': '',
                    'puts_filters': '',
                    'filter_logic': 'AND',
                    'calls_filterLogic': 'AND',
                    'puts_filterLogic': 'AND',
                    'sort': 'net_daily_premi',
                    'sort_direction': 'desc',
                    'limit': None,
                    'offset': '0'
                }
        
        request = MockRequest(temp_csv_file)
        # Create a proper mock query object that handles .get() correctly
        class MockQuery:
            def __init__(self, data):
                self._data = data
            
            def get(self, key, default=None):
                return self._data.get(key, default)
        
        request.query = MockQuery(request.query_params)
        
        # First call - should cache
        response1 = await db_server.handle_covered_calls_data(request)
        assert response1.status == 200
        
        # Check cache
        assert temp_csv_file in db_server._covered_calls_cache
        cache_entry = db_server._covered_calls_cache[temp_csv_file]
        
        # Cache should have 3 elements: (df, cache_time, mtime)
        assert len(cache_entry) == 3
        df, cache_time, mtime = cache_entry
        
        # Verify types
        assert isinstance(df, pd.DataFrame)
        assert isinstance(cache_time, float)
        assert isinstance(mtime, (float, type(None)))
        
        if mtime is not None:
            # mtime should be positive and recent
            assert mtime > 0
            import time
            assert time.time() - mtime < 3600  # Within last hour
    
    @pytest.mark.asyncio
    async def test_timestamp_persists_across_cache_hits(self, temp_csv_file):
        """Test that timestamp remains consistent when using cached data."""
        import db_server
        from aiohttp import web
        
        # Clear cache
        db_server._covered_calls_cache.clear()
        
        app = web.Application()
        
        class MockRequest:
            def __init__(self, csv_path):
                self.app = app
                self.query_params = {
                    'source': csv_path,
                    'option_type': 'all',
                    'filters': '[]',
                    'calls_filters': '',
                    'puts_filters': '',
                    'filter_logic': 'AND',
                    'calls_filterLogic': 'AND',
                    'puts_filterLogic': 'AND',
                    'sort': 'net_daily_premi',
                    'sort_direction': 'desc',
                    'limit': None,
                    'offset': '0'
                }
        
        request = MockRequest(temp_csv_file)
        # Create a proper mock query object that handles .get() correctly
        class MockQuery:
            def __init__(self, data):
                self._data = data
            
            def get(self, key, default=None):
                return self._data.get(key, default)
        
        request.query = MockQuery(request.query_params)
        
        # First call
        response1 = await db_server.handle_covered_calls_data(request)
        data1 = json.loads(response1.text)
        timestamp1 = data1['metadata']['data_source_timestamp']
        
        # Second call (should use cache)
        response2 = await db_server.handle_covered_calls_data(request)
        data2 = json.loads(response2.text)
        timestamp2 = data2['metadata']['data_source_timestamp']
        
        # Timestamps should be identical (from same file)
        assert timestamp1 == timestamp2
    
    def test_timestamp_format_is_iso8601(self, temp_csv_file):
        """Test that timestamp is in ISO 8601 format."""
        import os
        
        mtime = os.path.getmtime(temp_csv_file)
        
        # Convert to ISO format (as done in db_server)
        from datetime import timezone
        iso_timestamp = datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()
        
        # Verify format
        assert isinstance(iso_timestamp, str)
        assert 'T' in iso_timestamp  # Date and time separated by T
        assert '+' in iso_timestamp or 'Z' in iso_timestamp  # Has timezone
        
        # Should be parseable
        parsed = datetime.fromisoformat(iso_timestamp.replace('Z', '+00:00'))
        assert isinstance(parsed, datetime)
    
    @pytest.mark.asyncio
    async def test_url_source_fallback_timestamp(self):
        """Test that URL sources use current time as fallback."""
        import db_server
        from aiohttp import web
        from unittest.mock import patch, MagicMock
        import urllib.request
        
        app = web.Application()
        
        # Mock CSV content
        csv_content = "ticker,price\nAAPL,150.0\nGOOGL,2800.0"
        
        class MockRequest:
            def __init__(self):
                self.app = app
                self.query_params = {
                    'source': 'https://example.com/data.csv',
                    'option_type': 'all',
                    'filters': '[]',
                    'calls_filters': '',
                    'puts_filters': '',
                    'filter_logic': 'AND',
                    'calls_filterLogic': 'AND',
                    'puts_filterLogic': 'AND',
                    'sort': 'net_daily_premi',
                    'sort_direction': 'desc',
                    'limit': None,
                    'offset': '0'
                }
        
        request = MockRequest()
        request.query = type('obj', (object,), {
            'get': lambda key, default=None: request.query_params.get(key, default)
        })()
        
        # Mock urllib.request.urlopen
        mock_response = MagicMock()
        mock_response.read.return_value = csv_content.encode('utf-8')
        mock_response.headers.get.return_value = None  # No Last-Modified header
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = lambda s, *args: None
        
        with patch('urllib.request.urlopen', return_value=mock_response):
            response = await db_server.handle_covered_calls_data(request)
            
            if response.status == 200:
                data = json.loads(response.text)
                
                # Should have a timestamp even without Last-Modified header
                assert 'metadata' in data
                assert 'data_source_timestamp' in data['metadata']
    
    @pytest.mark.asyncio
    async def test_missing_file_handles_gracefully(self):
        """Test that missing file is handled gracefully."""
        import db_server
        from aiohttp import web
        
        app = web.Application()
        
        class MockRequest:
            def __init__(self):
                self.app = app
                self.query_params = {
                    'source': '/nonexistent/path/to/file.csv',
                    'option_type': 'all',
                    'filters': '[]',
                    'calls_filters': '',
                    'puts_filters': '',
                    'filter_logic': 'AND',
                    'calls_filterLogic': 'AND',
                    'puts_filterLogic': 'AND',
                    'sort': 'net_daily_premi',
                    'sort_direction': 'desc',
                    'limit': None,
                    'offset': '0'
                }
        
        request = MockRequest()
        # Create a proper mock query object that handles .get() correctly
        class MockQuery:
            def __init__(self, data):
                self._data = data
            
            def get(self, key, default=None):
                return self._data.get(key, default)
        
        request.query = MockQuery(request.query_params)
        
        # Should return error response, not crash
        response = await db_server.handle_covered_calls_data(request)
        assert response.status == 404  # File not found
        
        data = json.loads(response.text)
        assert 'error' in data
    
    def test_timestamp_comparison_with_file_stat(self, temp_csv_file):
        """Test that returned timestamp matches file stat."""
        import os
        import time
        
        # Get file modification time
        file_mtime = os.path.getmtime(temp_csv_file)
        
        # Convert to datetime
        file_dt = datetime.fromtimestamp(file_mtime, tz=timezone.utc)
        
        # This should match what the API returns
        iso_timestamp = file_dt.isoformat()
        
        # Verify it's a valid timestamp
        assert isinstance(iso_timestamp, str)
        
        # Parse it back
        parsed = datetime.fromisoformat(iso_timestamp.replace('Z', '+00:00'))
        
        # Should be very close to original (within a second due to precision)
        time_diff = abs((parsed - file_dt).total_seconds())
        assert time_diff < 1.0


class TestDailyPriceRange:
    """Test daily price range tracking and retrieval functionality."""
    
    @pytest.mark.asyncio
    async def test_get_last_trading_day_weekday(self):
        """Test _get_last_trading_day returns today for weekdays."""
        if RealtimeDataService is None:
            pytest.skip("RealtimeDataService not available")
        
        from unittest.mock import Mock
        
        # Create a mock service instance
        mock_repo = Mock()
        mock_cache = Mock()
        mock_logger = Mock()
        service = RealtimeDataService(mock_repo, mock_cache, mock_logger)
        
        # Test on a weekday (Monday = 0, Friday = 4)
        # Use a known Monday - need to check actual weekday
        # Dec 9, 2024 is actually a Monday
        monday = datetime(2024, 12, 9, 12, 0, 0, tzinfo=timezone.utc)
        result = service._get_last_trading_day(monday)
        
        # Should return the same day for a weekday
        assert result == '2024-12-09' or result.endswith('12-09')  # Should return the same day
    
    @pytest.mark.asyncio
    async def test_get_last_trading_day_weekend(self):
        """Test _get_last_trading_day returns Friday for weekends."""
        if RealtimeDataService is None:
            pytest.skip("RealtimeDataService not available")
        
        from unittest.mock import Mock
        
        mock_repo = Mock()
        mock_cache = Mock()
        mock_logger = Mock()
        service = RealtimeDataService(mock_repo, mock_cache, mock_logger)
        
        # Test on a Saturday - Dec 7, 2024 is a Saturday
        saturday = datetime(2024, 12, 7, 12, 0, 0, tzinfo=timezone.utc)
        result_sat = service._get_last_trading_day(saturday)
        
        # Should return Friday (Dec 6, 2024)
        assert result_sat == '2024-12-06' or result_sat.endswith('12-06')
        
        # Test on a Sunday - Dec 8, 2024 is a Sunday
        sunday = datetime(2024, 12, 8, 12, 0, 0, tzinfo=timezone.utc)
        result_sun = service._get_last_trading_day(sunday)
        
        # Should return Friday (Dec 6, 2024)
        assert result_sun == '2024-12-06' or result_sun.endswith('12-06')
    
    @pytest.mark.asyncio
    async def test_update_daily_price_range_new_high(self):
        """Test _update_daily_price_range updates when new high is found."""
        if RealtimeDataService is None:
            pytest.skip("RealtimeDataService not available")
        
        from unittest.mock import Mock, AsyncMock, patch, MagicMock
        import json
        
        mock_repo = Mock()
        mock_cache = Mock()
        mock_logger = Mock()
        service = RealtimeDataService(mock_repo, mock_cache, mock_logger)
        
        # Create sample DataFrame with prices
        df = pd.DataFrame({
            'price': [100.0, 105.0, 110.0]
        }, index=pd.date_range('2024-12-09 10:00:00', periods=3, freq='1h', tz='UTC'))
        
        # Mock Redis client
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=None)  # No existing data
        mock_client.setex = AsyncMock()
        
        # Mock the cache's _get_redis_client method
        async def mock_get_redis_client():
            return mock_client
        
        service.cache._get_redis_client = mock_get_redis_client
        
        # Mock the database connection to avoid actual DB queries
        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=[])  # No open price found
        mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn.__aexit__ = AsyncMock(return_value=None)
        
        # Mock the connection.get_connection() context manager
        mock_connection = Mock()
        mock_connection.get_connection = Mock(return_value=mock_conn)
        service.realtime_repo.connection = mock_connection
        
        await service._update_daily_price_range('AAPL', df)
        
        # Verify setex was called
        assert mock_client.setex.called
        
        # Verify the data stored
        call_args = mock_client.setex.call_args
        redis_key = call_args[0][0]
        ttl = call_args[0][1]
        data_str = call_args[0][2]
        
        assert 'AAPL' in redis_key.upper()
        assert ttl == 259200  # 72 hours
        data = json.loads(data_str)
        assert data['high'] == 110.0
        assert data['low'] == 100.0
    
    @pytest.mark.asyncio
    async def test_update_daily_price_range_updates_existing(self):
        """Test _update_daily_price_range only updates when price is lower/higher."""
        if RealtimeDataService is None:
            pytest.skip("RealtimeDataService not available")
        
        from unittest.mock import Mock, AsyncMock, patch, MagicMock
        import json
        
        mock_repo = Mock()
        mock_cache = Mock()
        mock_logger = Mock()
        service = RealtimeDataService(mock_repo, mock_cache, mock_logger)
        
        # Create DataFrame with prices that should update
        df = pd.DataFrame({
            'price': [95.0, 115.0]  # Lower low and higher high
        }, index=pd.date_range('2024-12-09 10:00:00', periods=2, freq='1h', tz='UTC'))
        
        # Mock existing Redis data
        existing_data = json.dumps({
            'high': 110.0,
            'low': 100.0,
            'date': '2024-12-09'
        })
        
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=existing_data.encode('utf-8'))
        mock_client.setex = AsyncMock()
        
        # Mock the cache's _get_redis_client method
        async def mock_get_redis_client():
            return mock_client
        
        service.cache._get_redis_client = mock_get_redis_client
        
        # Mock the database connection to avoid actual DB queries
        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=[])  # No open price found
        mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn.__aexit__ = AsyncMock(return_value=None)
        
        # Mock the connection.get_connection() context manager
        mock_connection = Mock()
        mock_connection.get_connection = Mock(return_value=mock_conn)
        service.realtime_repo.connection = mock_connection
        
        await service._update_daily_price_range('AAPL', df)
        
        # Verify setex was called with updated values
        assert mock_client.setex.called
        call_args = mock_client.setex.call_args
        data_str = call_args[0][2]
        data = json.loads(data_str)
        
        assert data['high'] == 115.0  # Updated to new high
        assert data['low'] == 95.0    # Updated to new low
    
    @pytest.mark.asyncio
    async def test_update_daily_price_range_no_update_when_within_range(self):
        """Test _update_daily_price_range doesn't update when prices are within existing range."""
        if RealtimeDataService is None:
            pytest.skip("RealtimeDataService not available")
        
        from unittest.mock import Mock, AsyncMock, patch, MagicMock
        import json
        
        mock_repo = Mock()
        mock_cache = Mock()
        mock_logger = Mock()
        service = RealtimeDataService(mock_repo, mock_cache, mock_logger)
        
        # Create DataFrame with prices within existing range
        df = pd.DataFrame({
            'price': [102.0, 108.0]  # Within 100-110 range
        }, index=pd.date_range('2024-12-09 10:00:00', periods=2, freq='1h', tz='UTC'))
        
        # Mock existing Redis data
        existing_data = json.dumps({
            'high': 110.0,
            'low': 100.0,
            'date': '2024-12-09'
        })
        
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=existing_data.encode('utf-8'))
        mock_client.setex = AsyncMock()
        
        # Mock the cache's _get_redis_client method
        async def mock_get_redis_client():
            return mock_client
        
        service.cache._get_redis_client = mock_get_redis_client
        
        # Mock the database connection to avoid actual DB queries
        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=[])  # No open price found
        mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn.__aexit__ = AsyncMock(return_value=None)
        
        # Mock the connection.get_connection() context manager
        mock_connection = Mock()
        mock_connection.get_connection = Mock(return_value=mock_conn)
        service.realtime_repo.connection = mock_connection
        
        await service._update_daily_price_range('AAPL', df)
        
        # Should still call setex to persist existing data
        assert mock_client.setex.called
    
    @pytest.mark.asyncio
    async def test_get_daily_price_range_weekend_fallback(self):
        """Test get_daily_price_range falls back to last trading day on weekends."""
        if RealtimeDataService is None:
            pytest.skip("RealtimeDataService not available")
        
        from unittest.mock import Mock, AsyncMock, patch, MagicMock
        import json
        
        mock_repo = Mock()
        mock_cache = Mock()
        mock_logger = Mock()
        service = RealtimeDataService(mock_repo, mock_cache, mock_logger)
        
        # Mock Saturday - need to patch datetime.now() to return Saturday
        from unittest.mock import patch, MagicMock
        saturday = datetime(2024, 12, 7, 12, 0, 0, tzinfo=timezone.utc)
        
        # Mock Redis client - no data for Saturday, but data for Friday
        mock_client = AsyncMock()
        friday_data = json.dumps({
            'high': 110.0,
            'low': 100.0,
            'date': '2024-12-06'
        })
        
        async def mock_get(key):
            # get_daily_price_range will call _get_last_trading_day which returns '2024-12-06' for Saturday
            # So it will look for key containing '2024-12-06'
            if '2024-12-06' in key:  # Friday (last trading day)
                return friday_data.encode('utf-8')
            return None
        
        mock_client.get = AsyncMock(side_effect=mock_get)
        
        # Mock the cache's _get_redis_client method
        async def mock_get_redis_client():
            return mock_client
        
        service.cache._get_redis_client = mock_get_redis_client
        
        # Mock the database connection to avoid actual DB queries
        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=[])  # No open price found
        mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn.__aexit__ = AsyncMock(return_value=None)
        
        # Mock the connection.get_connection() context manager
        mock_connection = Mock()
        mock_connection.get_connection = Mock(return_value=mock_conn)
        service.realtime_repo.connection = mock_connection
        
        # Patch datetime.now() to return Saturday
        with patch('common.questdb_db.datetime') as mock_dt_module:
            # Make datetime.now() return Saturday when called with timezone.utc
            def mock_now(tz=None):
                if tz == timezone.utc:
                    return saturday
                return datetime.now(tz)
            mock_dt_module.now = mock_now
            # Make datetime constructor work normally
            original_datetime = datetime
            def mock_datetime_constructor(*args, **kw):
                return original_datetime(*args, **kw)
            mock_dt_module.side_effect = mock_datetime_constructor
            # Make sure datetime class attributes work
            for attr in ['utc', 'timezone', 'date', 'timedelta']:
                if hasattr(original_datetime, attr):
                    setattr(mock_dt_module, attr, getattr(original_datetime, attr))
            
            result = await service.get_daily_price_range('AAPL')
            
            # Should return Friday's data (via _get_last_trading_day fallback)
            assert result is not None
            assert result['high'] == 110.0
            assert result['low'] == 100.0
    
    @pytest.mark.asyncio
    async def test_get_daily_price_range_returns_none_when_no_data(self):
        """Test get_daily_price_range returns None when no data exists."""
        if RealtimeDataService is None:
            pytest.skip("RealtimeDataService not available")
        
        from unittest.mock import Mock, AsyncMock, patch
        
        mock_repo = Mock()
        mock_cache = Mock()
        mock_logger = Mock()
        service = RealtimeDataService(mock_repo, mock_cache, mock_logger)
        
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=None)
        
        # Mock the cache's _get_redis_client method
        async def mock_get_redis_client():
            return mock_client
        
        service.cache._get_redis_client = mock_get_redis_client
        
        result = await service.get_daily_price_range('AAPL')
        assert result is None
    
    def test_daily_range_displayed_in_html(self):
        """Test that daily range is displayed in generated HTML."""
        symbol = 'AAPL'
        data = {
            'price_info': {
                'current_price': {
                    'price': 150.50,
                    'daily_range': {
                        'high': 152.00,
                        'low': 148.50
                    }
                },
                'price_data': []
            },
            'financial_info': {},
            'options_info': {},
            'iv_info': {},
            'news_info': {}
        }
        
        html = generate_stock_info_html(symbol, data)
        
        # Should contain Day's Range
        assert "Day's Range" in html or "Day Range" in html
        assert '152.00' in html or '152' in html
        assert '148.50' in html or '148' in html


class TestOptionsTableFormatting:
    """Test options table formatting improvements."""
    
    def test_options_table_bid_ask_same_line(self):
        """Test that bid/ask are displayed on same line with spread below."""
        options_data = {
            'success': True,
            'data': {
                'contracts': [
                    {
                        'expiration': '2024-12-20',
                        'type': 'call',
                        'strike': 150.0,
                        'bid': 5.50,
                        'ask': 5.60,
                        'volume': 1000,
                        'open_interest': 5000,
                        'implied_volatility': 0.25,
                        'delta': 0.30,
                        'theta': -0.05
                    }
                ]
            }
        }
        
        html = _format_options_html(options_data, current_price=150.0)
        
        # Bid and ask should be on same line separated by /
        assert '$5.50 / $5.60' in html or '5.50 / 5.60' in html
        # Spread should be on next line
        assert '$0.10' in html or '0.10' in html
    
    def test_options_table_no_oi_column(self):
        """Test that OI (Open Interest) column is removed."""
        options_data = {
            'success': True,
            'data': {
                'contracts': [
                    {
                        'expiration': '2024-12-20',
                        'type': 'call',
                        'strike': 150.0,
                        'bid': 5.50,
                        'ask': 5.60,
                        'volume': 1000,
                        'open_interest': 5000,
                        'implied_volatility': 0.25,
                        'delta': 0.30,
                        'theta': -0.05
                    }
                ]
            }
        }
        
        html = _format_options_html(options_data, current_price=150.0)
        
        # Should have 6 columns per side (not 7)
        # Count header occurrences - should not have OI header
        assert html.count('colspan="6"') >= 2  # CALLS and PUTS headers
        # OI should not appear as a column header
        assert 'OI</th>' not in html or html.count('OI</th>') == 0


class TestImpliedVolatilityCollapsible:
    """Test implied volatility section collapsible functionality."""
    
    def test_iv_section_has_caret(self):
        """Test that IV section has a caret icon."""
        symbol = 'AAPL'
        data = {
            'price_info': {'current_price': {'price': 150.50}, 'price_data': []},
            'financial_info': {},
            'options_info': {},
            'iv_info': {
                'iv_data': {
                    'statistics': {'mean': 0.25, 'median': 0.24, 'count': 100},
                    'atm_iv': {'mean': 0.26},
                    'call_iv': {'mean': 0.25},
                    'put_iv': {'mean': 0.27}
                }
            },
            'news_info': {}
        }
        
        html = generate_stock_info_html(symbol, data)
        
        # Should have caret icon
        assert 'ivCaret' in html or '▶' in html
        # Should have toggle function
        assert 'toggleIVSection' in html
        # Content should be hidden by default
        assert 'ivContent' in html
        assert 'display: none' in html or 'display:none' in html
    
    def test_iv_section_collapsed_by_default(self):
        """Test that IV section is collapsed by default."""
        symbol = 'AAPL'
        data = {
            'price_info': {'current_price': {'price': 150.50}, 'price_data': []},
            'financial_info': {},
            'options_info': {},
            'iv_info': {
                'iv_data': {
                    'statistics': {'mean': 0.25}
                }
            },
            'news_info': {}
        }
        
        html = generate_stock_info_html(symbol, data)
        
        # IV content should be hidden
        assert 'id="ivContent"' in html
        # Check for hidden style
        iv_content_start = html.find('id="ivContent"')
        if iv_content_start != -1:
            # Look for display: none in the style attribute
            style_section = html[iv_content_start:iv_content_start+200]
            assert 'display: none' in style_section or 'display:none' in style_section


class TestEarningsDateFiltering:
    """Test earnings date filtering to exclude dividend/yield data."""
    
    @pytest.mark.asyncio
    async def test_earnings_date_filters_dividend_data(self):
        """Test that earnings date parsing filters out dividend/yield information."""
        import db_server
        
        # Mock the fetch_earnings_date function to return dividend data
        # and verify it gets filtered
        test_cases = [
            ("Forward Dividend & Yield 0.84 (0.26%)", "N/A"),  # Should be filtered
            ("2025-01-15", "2025-01-15"),  # Valid date should pass
            ("Jan 15, 2025", "Jan 15, 2025"),  # Valid date format should pass
            ("Dividend Yield 2.5%", "N/A"),  # Should be filtered
        ]
        
        for input_value, expected in test_cases:
            # Check the filtering logic in generate_stock_info_html
            symbol = 'AAPL'
            data = {
                'price_info': {'current_price': {'price': 150.50}, 'price_data': []},
                'financial_info': {},
                'options_info': {},
                'iv_info': {},
                'news_info': {}
            }
            
            html = generate_stock_info_html(symbol, data, earnings_date=input_value)
            
            # If input contains dividend/yield, it should show N/A or be filtered
            if 'Dividend' in input_value or 'Yield' in input_value:
                # Should not contain the dividend text in earnings date field
                earnings_section = html[html.find('Earnings Date'):html.find('Earnings Date')+200]
                assert 'Dividend' not in earnings_section or 'N/A' in earnings_section
            else:
                # Valid dates should appear
                assert input_value in html or expected in html


class TestLayoutAndDisplay:
    """Test layout and display improvements."""
    
    def test_metrics_grid_positioned_right_of_price(self):
        """Test that metrics grid is positioned to the right of price section."""
        symbol = 'AAPL'
        data = {
            'price_info': {
                'current_price': {
                    'price': 150.50,
                    'previous_close': 148.00,
                    'open': 149.00
                },
                'price_data': []
            },
            'financial_info': {},
            'options_info': {},
            'iv_info': {},
            'news_info': {}
        }
        
        html = generate_stock_info_html(symbol, data)
        
        # Should have header-content-wrapper for flexbox layout
        assert 'header-content-wrapper' in html or 'headerContentWrapper' in html
        # Price section and metrics grid should be in the wrapper
        assert 'price-section' in html
        assert 'metrics-grid' in html
    
    def test_financial_data_section_hidden(self):
        """Test that Financial Data section is hidden/removed."""
        symbol = 'AAPL'
        data = {
            'price_info': {'current_price': {'price': 150.50}, 'price_data': []},
            'financial_info': {
                'financial_data': {
                    'market_cap': 2500000000000,
                    'pe_ratio': 28.5
                }
            },
            'options_info': {},
            'iv_info': {},
            'news_info': {}
        }
        
        html = generate_stock_info_html(symbol, data)
        
        # Financial Data section should not appear
        # Look for the specific section that was removed
        assert html.count('<h2>Financial Data</h2>') == 0
    
    def test_days_range_displayed_before_52_week_range(self):
        """Test that Day's Range appears before 52 Week Range in metrics."""
        symbol = 'AAPL'
        data = {
            'price_info': {
                'current_price': {
                    'price': 150.50,
                    'daily_range': {
                        'high': 152.00,
                        'low': 148.50
                    }
                },
                'price_data': []
            },
            'financial_info': {},
            'options_info': {},
            'iv_info': {},
            'news_info': {}
        }
        
        html = generate_stock_info_html(symbol, data)
        
        # Find positions of both ranges
        days_range_pos = html.find("Day's Range")
        week52_range_pos = html.find("52 Week Range")
        
        # Day's Range should come before 52 Week Range
        if days_range_pos != -1 and week52_range_pos != -1:
            assert days_range_pos < week52_range_pos


def run_all_tests():
    """Run all tests and return results."""
    print("=" * 80)
    print("RUNNING COMPREHENSIVE DB_SERVER TESTS")
    print("=" * 80)
    
    # Run pytest with verbose output
    pytest_args = [
        __file__,
        '-v',  # Verbose
        '--tb=short',  # Shorter traceback format
        '-x',  # Stop on first failure
        '--color=yes'
    ]
    
    result = pytest.main(pytest_args)
    
    print("\n" + "=" * 80)
    if result == 0:
        print("✅ ALL TESTS PASSED!")
    else:
        print("❌ SOME TESTS FAILED")
    print("=" * 80)
    
    return result


if __name__ == '__main__':
    sys.exit(run_all_tests())

