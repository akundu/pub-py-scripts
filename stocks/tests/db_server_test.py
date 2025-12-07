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
    from server.websocket_manager import WebSocketManager
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
        assert 'call' in result
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
        assert 'redis_enabled' in stats
        assert stats['redis_enabled'] is False  # We disabled it in fixture


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
        assert 'call' in html and 'put' in html
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

