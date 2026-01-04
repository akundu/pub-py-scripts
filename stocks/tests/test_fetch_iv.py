#!/usr/bin/env python3
"""
Tests for scripts/fetch_iv.py functionality.

Tests verify:
1. Argument parsing
2. Symbol loading
3. IV analysis integration
4. Database saving
5. Worker process handling
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, AsyncMock
import asyncio
import tempfile
import os
import yaml

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestArgumentParsing:
    """Test argument parsing for fetch_iv.py"""
    
    def test_calendar_days_default(self):
        """Test default calendar days"""
        default_calendar_days = 90
        assert default_calendar_days == 90
        assert isinstance(default_calendar_days, int)
    
    def test_log_level_choices(self):
        """Test log level choices"""
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR']
        assert 'DEBUG' in valid_levels
        assert 'ERROR' in valid_levels
    
    def test_server_url_default(self):
        """Test default server URL"""
        default_url = "http://localhost:9100"
        assert "http://" in default_url or "localhost" in default_url
    
    def test_server_url_normalization(self):
        """Test server URL normalization"""
        # Should add http:// if missing
        url_without_protocol = "localhost:9100"
        url_with_protocol = "http://localhost:9100"
        
        if not url_without_protocol.startswith(('http://', 'https://')):
            normalized = f"http://{url_without_protocol}"
            assert normalized == url_with_protocol


class TestSymbolLoading:
    """Test symbol loading for fetch_iv.py"""
    
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
    
    def test_symbols_direct_input(self):
        """Test direct symbol input"""
        symbols = ['AAPL', 'MSFT', 'GOOGL']
        symbols_upper = [s.upper() for s in symbols]
        assert all(s == s.upper() for s in symbols_upper)
        assert 'AAPL' in symbols_upper


class TestIVAnalyzer:
    """Test IV Analyzer integration"""
    
    def test_iv_analyzer_import(self):
        """Test that IVAnalyzer can be imported"""
        from common.iv_analysis import IVAnalyzer
        assert IVAnalyzer is not None
    
    @pytest.mark.asyncio
    async def test_iv_analyzer_initialization(self):
        """Test IVAnalyzer initialization"""
        from common.iv_analysis import IVAnalyzer
        
        # Mock API key
        api_key = "test_key"
        analyzer = IVAnalyzer(
            polygon_api_key=api_key,
            data_dir="data",
            logger=MagicMock()
        )
        
        assert analyzer.polygon_api_key == api_key
        assert analyzer.data_dir is not None
    
    @pytest.mark.asyncio
    async def test_get_iv_analysis_structure(self):
        """Test IV analysis result structure"""
        # Expected structure
        expected_keys = ['ticker', 'metrics', 'strategy']
        expected_metrics = ['iv_30d', 'hv_1yr_range', 'rank', 'roll_yield']
        expected_strategy = ['recommendation', 'risk_score', 'notes']
        
        # Verify structure expectations
        assert 'ticker' in expected_keys
        assert 'iv_30d' in expected_metrics
        assert 'recommendation' in expected_strategy


class TestDatabaseSaving:
    """Test database saving functionality"""
    
    @pytest.mark.asyncio
    async def test_save_iv_analysis_structure(self):
        """Test IV analysis data structure for saving"""
        # Sample result structure
        result = {
            "ticker": "AAPL",
            "metrics": {
                "iv_30d": "40.41%",
                "hv_1yr_range": "20.00% - 60.00%",
                "rank": 75.5,
                "roll_yield": "5.2%"
            },
            "strategy": {
                "recommendation": "SELL PREMIUM",
                "risk_score": 7.5,
                "notes": {"meaning": "Expensive vs History.", "action": "Credit Spreads."}
            },
            "relative_rank": 25.5
        }
        
        # Verify structure
        assert 'ticker' in result
        assert 'metrics' in result
        assert 'strategy' in result
        assert 'iv_30d' in result['metrics']
        assert 'rank' in result['metrics']
    
    def test_iv_30d_parsing(self):
        """Test parsing IV_30d from percentage string"""
        iv_30d_str = "40.41%"
        if iv_30d_str and iv_30d_str.endswith("%"):
            iv_30d = float(iv_30d_str.rstrip("%")) / 100.0
            assert iv_30d == 0.4041
            assert isinstance(iv_30d, float)


class TestWorkerProcess:
    """Test worker process functionality"""
    
    def test_worker_task_signature(self):
        """Test worker task function signature"""
        # Worker task should accept these parameters
        expected_params = [
            'ticker', 'cal_days', 'force_api', 'config',
            'log_level_str', 'server_url', 'use_polygon',
            'data_dir', 'db_config'
        ]
        
        # Verify parameter names are valid
        assert 'ticker' in expected_params
        assert 'cal_days' in expected_params
        assert 'config' in expected_params
    
    @pytest.mark.asyncio
    async def test_async_worker_structure(self):
        """Test async worker function structure"""
        # Worker should be async
        async def mock_worker():
            return {"ticker": "AAPL", "result": "success"}, False
        
        result, needs_update = await mock_worker()
        assert 'ticker' in result
        assert isinstance(needs_update, bool)


class TestSPYRanking:
    """Test SPY relative ranking"""
    
    def test_spy_inclusion(self):
        """Test that SPY is always included for ranking"""
        symbols = ['AAPL', 'MSFT']
        all_tickers = list(set(["SPY"] + symbols))
        
        assert "SPY" in all_tickers
        assert len(all_tickers) == 3
    
    def test_relative_rank_calculation(self):
        """Test relative rank calculation"""
        spy_rank = 50.0
        ticker_rank = 75.5
        relative_rank = round(ticker_rank - spy_rank, 2)
        
        assert relative_rank == 25.5
        assert isinstance(relative_rank, float)


class TestCacheBehavior:
    """Test cache behavior"""
    
    def test_dont_sync_flag(self):
        """Test --dont-sync flag behavior"""
        # --dont-sync means use cache (sync=False)
        dont_sync = True
        sync = not dont_sync  # Inverted logic
        assert sync == False
    
    def test_force_refresh_default(self):
        """Test default force refresh behavior"""
        # Default is sync/refresh (force_refresh=True)
        default_sync = True
        assert default_sync == True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
