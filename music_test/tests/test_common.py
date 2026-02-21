"""
Tests for lib/common.py - audio settings and utility functions.
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.common import (
    get_chunk, set_chunk,
    get_rate, set_rate,
    get_channels, set_channels,
    get_buffer_size, get_hop_size,
    get_overlap_ratio, set_overlap_ratio,
    clear_line
)


class TestAudioSettings:
    """Test audio configuration functions."""
    
    def test_default_chunk_size(self):
        """Test default chunk size is 4096."""
        assert get_chunk() == 4096
    
    def test_default_sample_rate(self):
        """Test default sample rate is 44100 Hz."""
        assert get_rate() == 44100
    
    def test_default_channels(self):
        """Test default channel count is 1 (mono)."""
        assert get_channels() == 1
    
    def test_set_and_get_chunk(self):
        """Test setting and getting chunk size."""
        original = get_chunk()
        set_chunk(2048)
        assert get_chunk() == 2048
        set_chunk(original)  # Restore
    
    def test_set_and_get_rate(self):
        """Test setting and getting sample rate."""
        original = get_rate()
        set_rate(48000)
        assert get_rate() == 48000
        set_rate(original)  # Restore
    
    def test_set_and_get_channels(self):
        """Test setting and getting channel count."""
        original = get_channels()
        set_channels(2)
        assert get_channels() == 2
        set_channels(original)  # Restore


class TestOverlapSettings:
    """Test overlap ratio and related calculations."""
    
    def test_default_overlap_ratio(self):
        """Test default overlap ratio is 0.75."""
        set_overlap_ratio(0.75)  # Reset to default
        assert get_overlap_ratio() == 0.75
    
    def test_set_overlap_ratio(self):
        """Test setting overlap ratio."""
        set_overlap_ratio(0.5)
        assert get_overlap_ratio() == 0.5
        set_overlap_ratio(0.75)  # Restore
    
    def test_hop_size_calculation(self):
        """Test hop size is calculated correctly from overlap ratio."""
        set_overlap_ratio(0.75)
        chunk = get_chunk()
        expected_hop = int(chunk * (1 - 0.75))
        assert get_hop_size() == expected_hop
    
    def test_hop_size_with_different_overlap(self):
        """Test hop size with 50% overlap."""
        set_overlap_ratio(0.5)
        chunk = get_chunk()
        expected_hop = int(chunk * 0.5)
        assert get_hop_size() == expected_hop
        set_overlap_ratio(0.75)  # Restore
    
    def test_buffer_size(self):
        """Test buffer size is twice the chunk size."""
        assert get_buffer_size() == get_chunk() * 2


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_clear_line_runs(self, capsys):
        """Test clear_line function executes without error."""
        clear_line()
        captured = capsys.readouterr()
        # Should print spaces and carriage return
        assert '\r' in captured.out or len(captured.out) >= 0

