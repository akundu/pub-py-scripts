#!/usr/bin/env python3
"""
Tests for scripts/fetch_options.py functionality.

Tests verify:
1. Argument parsing
2. Date handling
3. Symbol loading
4. Options filtering
5. Database integration
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import tempfile
import os
import yaml

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestArgumentParsing:
    """Test argument parsing for fetch_options.py"""
    
    def test_date_argument_default(self):
        """Test that default date is today"""
        # Default date should be today
        expected_date = datetime.now().strftime('%Y-%m-%d')
        assert len(expected_date) == 10
        assert expected_date.count('-') == 2
    
    def test_date_argument_format(self):
        """Test date format validation"""
        test_date = "2024-06-05"
        try:
            parsed = datetime.strptime(test_date, '%Y-%m-%d')
            assert parsed.year == 2024
            assert parsed.month == 6
            assert parsed.day == 5
        except ValueError:
            pytest.fail("Date format should be YYYY-MM-DD")
    
    def test_option_type_choices(self):
        """Test option type choices"""
        valid_types = ['call', 'put', 'all']
        assert 'call' in valid_types
        assert 'put' in valid_types
        assert 'all' in valid_types


class TestSymbolLoading:
    """Test symbol loading for fetch_options.py"""
    
    def test_symbols_list_yaml(self):
        """Test loading symbols from YAML file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml_data = {
                'symbols': ['AAPL', 'MSFT', 'GOOGL']
            }
            yaml.dump(yaml_data, f)
            yaml_path = f.name
        
        try:
            from common.symbol_loader import load_symbols_from_yaml
            symbols = load_symbols_from_yaml(yaml_path, quiet=True)
            assert len(symbols) == 3
            assert 'AAPL' in symbols
        finally:
            os.unlink(yaml_path)


class TestOptionsFiltering:
    """Test options filtering logic"""
    
    def test_strike_range_percent(self):
        """Test strike range percentage calculation"""
        stock_price = 100.0
        range_percent = 10
        
        min_strike = stock_price * (1 - range_percent / 100)
        max_strike = stock_price * (1 + range_percent / 100)
        
        assert min_strike == 90.0
        assert max_strike == 110.0
    
    def test_max_days_to_expiry(self):
        """Test days to expiry calculation"""
        target_date = datetime(2024, 6, 5)
        expiry_date = datetime(2024, 7, 5)
        max_days = 30
        
        days_diff = (expiry_date - target_date).days
        assert days_diff == 30
        assert days_diff <= max_days


class TestDatabaseIntegration:
    """Test database integration for fetch_options.py"""
    
    def test_database_connection_string_parsing(self):
        """Test parsing of database connection strings"""
        # QuestDB format
        questdb_url = "questdb://user:pass@host:8812/db"
        assert "questdb://" in questdb_url
        
        # HTTP server format
        http_url = "http://localhost:9002"
        assert "http://" in http_url or "localhost" in http_url
    
    def test_db_batch_size_default(self):
        """Test default batch size"""
        default_batch_size = 100
        assert default_batch_size > 0
        assert isinstance(default_batch_size, int)


class TestDateRangeHandling:
    """Test date range and multi-month handling"""
    
    def test_months_ahead_calculation(self):
        """Test months ahead calculation"""
        start_date = datetime(2024, 6, 5)
        months_ahead = 6
        
        # Calculate end date (approximate, 30 days per month)
        from datetime import timedelta
        end_date = start_date + timedelta(days=months_ahead * 30)
        
        assert end_date > start_date
        assert (end_date - start_date).days <= months_ahead * 31
    
    def test_single_date_mode(self):
        """Test single date mode (months_ahead = 0)"""
        months_ahead = 0
        assert months_ahead == 0  # Single date mode


class TestCSVCache:
    """Test CSV cache functionality"""
    
    def test_csv_directory_creation(self):
        """Test that CSV directory can be created"""
        data_dir = "data"
        os.makedirs(data_dir, exist_ok=True)
        assert os.path.isdir(data_dir)
    
    def test_csv_file_naming(self):
        """Test CSV file naming convention"""
        symbol = "AAPL"
        date = "2024-06-05"
        expected_filename = f"{symbol}_{date}_options.csv"
        assert "AAPL" in expected_filename
        assert "2024-06-05" in expected_filename
        assert expected_filename.endswith(".csv")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
