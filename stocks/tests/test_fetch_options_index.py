"""
Tests for index symbol handling in fetch_options.py.

Verifies that I:SPX is correctly converted to SPX for Polygon options API.
"""

import pytest
import sys
from pathlib import Path

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from fetch_options import _convert_index_symbol_for_polygon


class TestIndexSymbolConversion:
    """Test index symbol conversion for Polygon API."""
    
    def test_convert_index_symbol_i_spx(self):
        """Test that I:SPX converts to SPX."""
        polygon_symbol, is_index = _convert_index_symbol_for_polygon("I:SPX")
        assert polygon_symbol == "SPX"
        assert is_index is True
    
    def test_convert_index_symbol_i_ndx(self):
        """Test that I:NDX converts to NDX."""
        polygon_symbol, is_index = _convert_index_symbol_for_polygon("I:NDX")
        assert polygon_symbol == "NDX"
        assert is_index is True
    
    def test_convert_regular_symbol(self):
        """Test that regular symbols are unchanged."""
        polygon_symbol, is_index = _convert_index_symbol_for_polygon("AAPL")
        assert polygon_symbol == "AAPL"
        assert is_index is False
    
    def test_convert_lowercase_index(self):
        """Test that lowercase index symbols are converted."""
        polygon_symbol, is_index = _convert_index_symbol_for_polygon("i:spx")
        assert polygon_symbol == "SPX"
        assert is_index is True
    
    def test_convert_mixed_case(self):
        """Test that mixed case symbols are normalized."""
        polygon_symbol, is_index = _convert_index_symbol_for_polygon("I:spx")
        assert polygon_symbol == "SPX"
        assert is_index is True


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
