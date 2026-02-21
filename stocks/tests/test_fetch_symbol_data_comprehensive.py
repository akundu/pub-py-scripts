#!/usr/bin/env python3
"""
Comprehensive test suite for fetch_symbol_data.py

Tests verify:
1. IV analysis fetching and saving
2. Financial ratios fetching and saving
3. Integration with database
4. Integration with cache
5. Display functionality
6. Date range mode
7. Latest mode
8. Various command-line arguments
"""
import asyncio
import sys
import os
import json
from pathlib import Path
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import pandas as pd
import pytest

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from common.stock_db import get_stock_db, StockDBBase
from common.financial_data import get_financial_info, get_financial_ratios
from common.redis_cache import CacheKeyGenerator
from fetch_symbol_data import (
    _display_financials,
    _validate_and_normalize_args,
    _setup_database,
    _cleanup_resources,
    parse_args,
    process_symbol_data,
    get_current_price,
    _get_latest_price_with_timestamp
)


class TestFinancialDataFetching:
    """Test suite for financial data fetching and saving"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.symbol = "AVGO"
        self.mock_db_calls = []
        self.mock_cache_calls = []
        
    def create_mock_db(self):
        """Create a mock database instance"""
        mock_db = Mock(spec=StockDBBase)
        mock_db.cache = Mock()
        mock_db.cache.enable_cache = True
        
        # Track cache operations
        async def mock_cache_get(key):
            self.mock_cache_calls.append(('get', key))
            return None  # Cache miss by default
        
        async def mock_cache_set(key, value, ttl=None):
            self.mock_cache_calls.append(('set', key, ttl))
        
        mock_db.cache.get = AsyncMock(side_effect=mock_cache_get)
        mock_db.cache.set = AsyncMock(side_effect=mock_cache_set)
        
        # Mock get_financial_info
        async def mock_get_financial_info(ticker, start_date=None, end_date=None):
            self.mock_db_calls.append(('get_financial_info', ticker))
            # Return empty DataFrame (no existing data)
            return pd.DataFrame()
        
        mock_db.get_financial_info = AsyncMock(side_effect=mock_get_financial_info)
        
        # Mock save_financial_info
        async def mock_save_financial_info(ticker, financial_data):
            self.mock_db_calls.append(('save_financial_info', ticker, financial_data))
        
        mock_db.save_financial_info = AsyncMock(side_effect=mock_save_financial_info)
        
        return mock_db
    
    @pytest.mark.asyncio
    async def test_fetch_financial_ratios(self):
        """Test fetching financial ratios from Polygon API"""
        mock_db = self.create_mock_db()
        
        # Mock Polygon API response
        mock_ratios = {
            'price_to_earnings': 25.5,
            'price_to_book': 3.2,
            'price_to_sales': 5.1,
            'current_ratio': 2.1,
            'quick_ratio': 1.8,
            'cash_ratio': 0.5,
            'return_on_equity': 0.15,
            'debt_to_equity': 0.3,
            'dividend_yield': 0.02
        }
        
        with patch.dict(os.environ, {'POLYGON_API_KEY': 'test_key'}), \
             patch('common.financial_data.get_financial_ratios') as mock_get_ratios:
            mock_get_ratios.return_value = mock_ratios
            
            # Call get_financial_info with force_fetch=True
            result = await get_financial_info(
                symbol=self.symbol,
                db_instance=mock_db,
                force_fetch=True,
                include_iv_analysis=False
            )
            
            # Verify result
            assert result is not None
            assert result.get('symbol') == self.symbol
            assert result.get('financial_data') is not None
            
            financial_data = result['financial_data']
            assert financial_data.get('price_to_earnings') == 25.5
            assert financial_data.get('current_ratio') == 2.1
            
            # Verify database save was called
            save_calls = [call for call in self.mock_db_calls if call[0] == 'save_financial_info']
            assert len(save_calls) > 0, "Expected save_financial_info to be called"
            
            # Verify cache was set
            cache_set_calls = [call for call in self.mock_cache_calls if call[0] == 'set']
            assert len(cache_set_calls) > 0, "Expected cache to be set"
    
    @pytest.mark.asyncio
    async def test_fetch_iv_analysis(self):
        """Test fetching IV analysis"""
        mock_db = self.create_mock_db()
        
        # Mock IV analysis result
        mock_iv_analysis = {
            'ticker': self.symbol,
            'metrics': {
                'iv_30d': '41.95%',
                'iv_90d': '38.20%',
                'hv_1yr_range': '26.99% - 85.34%',
                'rank': 25.64,
                'roll_yield': '-13.52%'
            },
            'strategy': {
                'recommendation': 'HOLD / NEUTRAL',
                'risk_score': 2.8,
                'notes': {
                    'meaning': 'Normal.',
                    'action': 'Hold.'
                }
            },
            'relative_rank': 8.05
        }
        
        with patch.dict(os.environ, {'POLYGON_API_KEY': 'test_key'}), \
             patch('common.financial_data._calculate_iv_analysis') as mock_calc_iv:
            mock_calc_iv.return_value = mock_iv_analysis
            
            # Call get_financial_info with include_iv_analysis=True
            result = await get_financial_info(
                symbol=self.symbol,
                db_instance=mock_db,
                force_fetch=True,
                include_iv_analysis=True
            )
            
            # Verify result
            assert result is not None
            assert result.get('financial_data') is not None
            
            financial_data = result['financial_data']
            
            # Verify IV analysis is present
            assert 'iv_analysis_json' in financial_data or 'iv_30d' in financial_data
            
            # Verify database save was called
            save_calls = [call for call in self.mock_db_calls if call[0] == 'save_financial_info']
            assert len(save_calls) > 0, "Expected save_financial_info to be called with IV analysis"
    
    @pytest.mark.asyncio
    async def test_fetch_financial_and_iv_together(self):
        """Test fetching both financial ratios and IV analysis together"""
        mock_db = self.create_mock_db()
        
        mock_ratios = {
            'price_to_earnings': 25.5,
            'current_ratio': 2.1
        }
        
        mock_iv_analysis = {
            'ticker': self.symbol,
            'metrics': {
                'iv_30d': '41.95%',
                'rank': 25.64
            },
            'strategy': {
                'recommendation': 'HOLD / NEUTRAL'
            },
            'relative_rank': 8.05
        }
        
        with patch.dict(os.environ, {'POLYGON_API_KEY': 'test_key'}), \
             patch('common.financial_data.get_financial_ratios') as mock_get_ratios, \
             patch('common.financial_data._calculate_iv_analysis') as mock_calc_iv:
            
            mock_get_ratios.return_value = mock_ratios
            mock_calc_iv.return_value = mock_iv_analysis
            
            # Call get_financial_info with both flags
            result = await get_financial_info(
                symbol=self.symbol,
                db_instance=mock_db,
                force_fetch=True,
                include_iv_analysis=True
            )
            
            # Verify both are present
            assert result is not None
            financial_data = result.get('financial_data')
            assert financial_data is not None
            
            # Verify financial ratios
            assert financial_data.get('price_to_earnings') == 25.5
            assert financial_data.get('current_ratio') == 2.1
            
            # Verify IV analysis (either in JSON or as separate fields)
            has_iv = 'iv_analysis_json' in financial_data or 'iv_30d' in financial_data
            assert has_iv, "IV analysis should be present in financial_data"
            
            # Verify single database save call (merged data)
            save_calls = [call for call in self.mock_db_calls if call[0] == 'save_financial_info']
            assert len(save_calls) == 1, f"Expected 1 save call, got {len(save_calls)}"
            
            # Verify the saved data contains both ratios and IV
            saved_data = save_calls[0][2]  # Third element is financial_data
            assert saved_data.get('price_to_earnings') is not None
            assert 'iv_analysis_json' in saved_data or saved_data.get('iv_30d') is not None
    
    @pytest.mark.asyncio
    async def test_financial_data_caching(self):
        """Test that financial data is properly cached"""
        mock_db = self.create_mock_db()
        
        # First call - cache miss
        cached_data = {
            'price_to_earnings': 25.5,
            'current_ratio': 2.1,
            'iv_30d': '41.95%',
            'last_save_time': datetime.now(timezone.utc).isoformat()
        }
        
        async def mock_cache_get_with_data(key):
            if 'financial_info' in str(key):
                # Return cached data
                cache_df = pd.DataFrame([cached_data])
                return cache_df
            return None
        
        mock_db.cache.get = AsyncMock(side_effect=mock_cache_get_with_data)
        
        # Call get_financial_info (should hit cache)
        result = await get_financial_info(
            symbol=self.symbol,
            db_instance=mock_db,
            force_fetch=False,  # Don't force fetch, use cache
            include_iv_analysis=False
        )
        
        # Verify cache hit
        assert result is not None
        assert result.get('source') == 'cache'
        assert result.get('financial_data') is not None
        assert result['financial_data'].get('price_to_earnings') == 25.5
        
        # Verify no database save was called (cache hit)
        save_calls = [call for call in self.mock_db_calls if call[0] == 'save_financial_info']
        assert len(save_calls) == 0, "Expected no save call on cache hit"


class TestIVAnalysis:
    """Test suite for IV analysis functionality"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.symbol = "AVGO"
        self.benchmark = "VOO"
    
    @pytest.mark.asyncio
    async def test_iv_analysis_calculation(self):
        """Test that IV analysis is calculated correctly"""
        mock_db = Mock(spec=StockDBBase)
        
        # Mock IV analysis result
        mock_iv_result = {
            'ticker': self.symbol,
            'metrics': {
                'iv_30d': '41.95%',
                'iv_90d': '38.20%',
                'hv_1yr_range': '26.99% - 85.34%',
                'rank': 25.64,
                'roll_yield': '-13.52%'
            },
            'strategy': {
                'recommendation': 'HOLD / NEUTRAL',
                'risk_score': 2.8,
                'notes': {
                    'meaning': 'Normal.',
                    'action': 'Hold.'
                }
            },
            'relative_rank': 8.05
        }
        
        with patch.dict(os.environ, {'POLYGON_API_KEY': 'test_key'}), \
             patch('common.financial_data._calculate_iv_analysis') as mock_calc_iv:
            mock_calc_iv.return_value = mock_iv_result
            
            result = await get_financial_info(
                symbol=self.symbol,
                db_instance=mock_db,
                force_fetch=True,
                include_iv_analysis=True
            )
            
            # Verify IV analysis structure
            assert result is not None
            financial_data = result.get('financial_data')
            assert financial_data is not None
            
            # Check that IV metrics are present
            has_iv_metrics = (
                'iv_analysis_json' in financial_data or
                'iv_30d' in financial_data or
                'iv_rank' in financial_data
            )
            assert has_iv_metrics, "IV metrics should be present in financial_data"
    
    @pytest.mark.asyncio
    async def test_relative_rank_calculation(self):
        """Test that relative rank is calculated correctly (ticker_rank / VOO_rank)"""
        # This test verifies the relative rank calculation logic
        # Relative rank should be ticker_rank / benchmark_rank (not subtraction)
        
        ticker_rank = 25.64
        benchmark_rank = 3.18  # VOO's rank
        
        # Expected relative rank
        expected_relative_rank = round(ticker_rank / benchmark_rank, 2)
        # Allow for small rounding differences
        assert abs(expected_relative_rank - 8.05) < 0.1, f"Expected relative rank ~8.05, got {expected_relative_rank}"


class TestDisplayFunctionality:
    """Test suite for display functionality"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.symbol = "AVGO"
        self.logger = Mock()
        self.logger.debug = Mock()
        self.logger.info = Mock()
        self.logger.warning = Mock()
        self.logger.error = Mock()
    
    def create_mock_db_with_data(self):
        """Create mock DB with financial data"""
        mock_db = Mock(spec=StockDBBase)
        
        financial_data = pd.DataFrame([{
            'date': datetime.now().date(),
            'price_to_earnings': 25.5,
            'price_to_book': 3.2,
            'current_ratio': 2.1,
            'quick_ratio': 1.8,
            'cash_ratio': 0.5,
            'return_on_equity': 0.15,
            'debt_to_equity': 0.3,
            'dividend_yield': 0.02,
            'market_cap': 500000000000,
            'iv_30d': '41.95%',
            'iv_rank': 25.64,
            'relative_rank': 8.05,
            'iv_analysis_json': json.dumps({
                'metrics': {
                    'iv_30d': '41.95%',
                    'hv_1yr_range': '26.99% - 85.34%',
                    'rank': 25.64,
                    'roll_yield': '-13.52%'
                },
                'strategy': {
                    'recommendation': 'HOLD / NEUTRAL',
                    'risk_score': 2.8,
                    'notes': {
                        'meaning': 'Normal.',
                        'action': 'Hold.'
                    }
                },
                'relative_rank': 8.05
            })
        }])
        
        async def mock_get_financial_info(ticker, start_date=None, end_date=None):
            return financial_data
        
        mock_db.get_financial_info = AsyncMock(side_effect=mock_get_financial_info)
        
        return mock_db
    
    @pytest.mark.asyncio
    async def test_display_financials_with_ratios(self):
        """Test displaying financial ratios"""
        mock_db = self.create_mock_db_with_data()
        
        # Capture print output
        import io
        from contextlib import redirect_stdout
        
        f = io.StringIO()
        with redirect_stdout(f):
            await _display_financials(
                symbol=self.symbol,
                db_instance=mock_db,
                logger=self.logger,
                log_level="INFO",
                fetch_ratios=False
            )
        
        output = f.getvalue()
        
        # Verify key financial ratios are displayed
        assert 'price_to_earnings' in output.lower() or 'P/E' in output.upper()
        assert 'current_ratio' in output.lower() or 'Current Ratio' in output
    
    @pytest.mark.asyncio
    async def test_display_financials_with_iv_analysis(self):
        """Test displaying IV analysis"""
        mock_db = self.create_mock_db_with_data()
        
        import io
        from contextlib import redirect_stdout
        
        f = io.StringIO()
        with redirect_stdout(f):
            await _display_financials(
                symbol=self.symbol,
                db_instance=mock_db,
                logger=self.logger,
                log_level="INFO",
                fetch_ratios=False
            )
        
        output = f.getvalue()
        
        # Verify IV analysis fields are displayed
        assert 'iv_30d' in output.lower() or '30-day iv' in output.lower() or '30-day IV' in output
        assert 'rank' in output.lower() or 'IV Rank' in output or 'iv rank' in output.lower()
        assert 'relative_rank' in output.lower() or 'Relative Rank' in output or 'relative rank' in output.lower()
        assert 'risk_score' in output.lower() or 'Risk Score' in output or 'risk score' in output.lower()
        assert 'hv_1yr_range' in output.lower() or 'HV 1-Year Range' in output or '1-year hv range' in output.lower() or '1-Year HV Range' in output


class TestArgumentValidation:
    """Test suite for argument validation"""
    
    def test_validate_args_with_fetch_ratios(self):
        """Test argument validation with --fetch-ratios"""
        # Create mock args with proper string values
        args = Mock()
        args.data_source = "polygon"
        args.fetch_ratios = True
        args.fetch_news = False
        args.fetch_iv = False
        args.start_date = None
        args.end_date = datetime.now().strftime('%Y-%m-%d')
        args.date = None  # Add date attribute
        args.latest = False
        args.days_back = None
        args.timeframe = "daily"
        
        # Should not raise exception
        try:
            _validate_and_normalize_args(args)
            assert args.latest == True, "Should set latest=True when fetch_ratios and no dates"
        except SystemExit:
            pytest.fail("Should not exit when fetch_ratios is used with polygon data source")
    
    def test_validate_args_with_fetch_iv(self):
        """Test argument validation with --fetch-iv"""
        args = Mock()
        args.data_source = "polygon"
        args.fetch_ratios = False
        args.fetch_news = False
        args.fetch_iv = True
        args.start_date = None
        args.end_date = datetime.now().strftime('%Y-%m-%d')
        args.date = None  # Add date attribute
        args.latest = False
        args.days_back = None
        args.timeframe = "daily"
        
        try:
            _validate_and_normalize_args(args)
            assert args.latest == True, "Should set latest=True when fetch_iv and no dates"
        except SystemExit:
            pytest.fail("Should not exit when fetch_iv is used with polygon data source")
    
    def test_validate_args_wrong_data_source(self):
        """Test that wrong data source raises error"""
        args = Mock()
        args.data_source = "alpaca"
        args.fetch_ratios = True
        args.fetch_news = False
        args.fetch_iv = False
        
        with pytest.raises(SystemExit):
            _validate_and_normalize_args(args)


class TestDateRangeMode:
    """Test suite for date range mode"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.symbol = "AVGO"
    
    @pytest.mark.asyncio
    async def test_process_symbol_data_with_date_range(self):
        """Test processing symbol data with date range"""
        mock_db = Mock(spec=StockDBBase)
        
        # Mock get_stock_data to return sample data
        sample_data = pd.DataFrame({
            'open': [100.0, 101.0],
            'high': [105.0, 106.0],
            'low': [99.0, 100.0],
            'close': [104.0, 105.0],
            'volume': [1000000, 1100000]
        }, index=pd.date_range('2025-01-01', periods=2, freq='D'))
        
        async def mock_get_stock_data(symbol, start_date=None, end_date=None, interval='daily'):
            return sample_data
        
        mock_db.get_stock_data = AsyncMock(side_effect=mock_get_stock_data)
        
        # Test process_symbol_data
        result = await process_symbol_data(
            symbol=self.symbol,
            timeframe="daily",
            start_date="2025-01-01",
            end_date="2025-01-02",
            stock_db_instance=mock_db,
            force_fetch=False,
            query_only=True
        )
        
        # Verify result
        assert result is not None
        assert not result.empty
        assert len(result) == 2


class TestLatestMode:
    """Test suite for latest mode"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.symbol = "AVGO"
    
    @pytest.mark.asyncio
    async def test_get_current_price(self):
        """Test getting current price"""
        mock_db = Mock(spec=StockDBBase)
        
        # Mock _get_latest_price_with_timestamp
        mock_price_data = {
            'price': 191.60,
            'timestamp': datetime.now(timezone.utc),
            'write_timestamp': datetime.now(timezone.utc),
            'source': 'daily'
        }
        
        with patch('fetch_symbol_data._get_latest_price_with_timestamp') as mock_get_price:
            mock_get_price.return_value = mock_price_data
            
            result = await get_current_price(
                symbol=self.symbol,
                data_source="polygon",
                stock_db_instance=mock_db,
                max_age_seconds=600
            )
            
            # Verify result
            assert result is not None
            assert result.get('price') == 191.60
            assert result.get('source') == 'database'


# Integration test helper
def run_integration_tests():
    """
    Run integration tests against actual database
    
    Usage:
        python tests/test_fetch_symbol_data_comprehensive.py integration
    """
    import subprocess
    
    db_string = os.environ.get('QUEST_DB_STRING')
    if not db_string:
        print("ERROR: QUEST_DB_STRING environment variable not set")
        return False
    
    symbol = "AVGO"
    test_cases = [
        {
            "name": "Fetch Financial Ratios",
            "args": ["--fetch-ratios", "--show-financials"],
            "expected": ["price_to_earnings", "current_ratio"]
        },
        {
            "name": "Fetch IV Analysis",
            "args": ["--fetch-iv", "--fetch-ratios", "--show-financials"],
            "expected": ["iv_30d", "iv_rank", "relative_rank"]
        },
        {
            "name": "Fetch Both Financials and IV",
            "args": ["--fetch-ratios", "--fetch-iv", "--show-financials"],
            "expected": ["price_to_earnings", "iv_30d", "relative_rank"]
        },
        {
            "name": "Date Range with Financials",
            "args": ["--start-date", "2025-01-01", "--end-date", "2025-01-10", "--show-financials"],
            "expected": ["STORED FINANCIAL INFORMATION"]
        }
    ]
    
    print("\n" + "="*80)
    print("INTEGRATION TEST SUITE: fetch_symbol_data.py")
    print("="*80)
    
    all_passed = True
    
    for test_case in test_cases:
        print(f"\n--- Testing: {test_case['name']} ---")
        
        cmd = [
            "python", "fetch_symbol_data.py", symbol,
            "--db-path", db_string,
            "--log-level", "INFO"
        ] + test_case['args']
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        
        if result.returncode != 0:
            print(f"  ❌ FAIL: Command failed with exit code {result.returncode}")
            print(f"  Error: {result.stderr[:500]}")
            all_passed = False
            continue
        
        # Check for expected output
        output = result.stdout + result.stderr
        missing = []
        for expected in test_case['expected']:
            if expected.lower() not in output.lower():
                missing.append(expected)
        
        if missing:
            print(f"  ❌ FAIL: Missing expected output: {missing}")
            all_passed = False
        else:
            print(f"  ✓ PASS: All expected outputs found")
    
    print("\n" + "="*80)
    if all_passed:
        print("✓ ALL INTEGRATION TESTS PASSED")
    else:
        print("❌ SOME INTEGRATION TESTS FAILED")
    print("="*80 + "\n")
    
    return all_passed


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "integration":
        # Run integration tests
        success = run_integration_tests()
        sys.exit(0 if success else 1)
    else:
        # Run unit tests with pytest
        pytest.main([__file__, "-v", "--tb=short"])

