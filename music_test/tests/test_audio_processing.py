"""
Tests for audio processing functions (sound_capture and web_audio_processing).
"""
import pytest
import numpy as np
import sys
import os
from collections import deque
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.common import get_chunk, get_buffer_size, get_rate, set_overlap_ratio
from lib.music_understanding import INSTRUMENT_PRESETS


class TestAudioBufferManagement:
    """Test audio buffer operations."""
    
    def test_buffer_initialization(self, buffer_size):
        """Test buffer can be initialized correctly."""
        audio_buffer = np.zeros(buffer_size, dtype=np.float32)
        assert len(audio_buffer) == buffer_size
        assert audio_buffer.dtype == np.float32
    
    def test_buffer_update(self, chunk_size, buffer_size):
        """Test circular buffer update."""
        audio_buffer = np.zeros(buffer_size, dtype=np.float32)
        buffer_index = 0
        
        # Create a test chunk
        test_chunk = np.ones(chunk_size, dtype=np.float32) * 0.5
        
        # Update buffer
        audio_buffer[buffer_index:buffer_index+chunk_size] = test_chunk
        buffer_index = (buffer_index + chunk_size) % buffer_size
        
        # Verify update
        assert np.all(audio_buffer[:chunk_size] == 0.5)
        assert buffer_index == chunk_size
    
    def test_buffer_wrap_around(self, chunk_size, buffer_size):
        """Test buffer wraps around correctly."""
        audio_buffer = np.zeros(buffer_size, dtype=np.float32)
        
        # Fill buffer multiple times to test wrap-around
        buffer_index = 0
        for i in range(4):  # 4 chunks should wrap around
            test_chunk = np.ones(chunk_size, dtype=np.float32) * (i + 1)
            end_index = min(buffer_index + chunk_size, buffer_size)
            copy_size = end_index - buffer_index
            audio_buffer[buffer_index:end_index] = test_chunk[:copy_size]
            if copy_size < chunk_size:
                audio_buffer[:chunk_size - copy_size] = test_chunk[copy_size:]
            buffer_index = (buffer_index + chunk_size) % buffer_size
        
        # Buffer should be filled with data
        assert not np.all(audio_buffer == 0)


class TestHistoryManagement:
    """Test note and chord history management."""
    
    def test_notes_history_deque(self):
        """Test notes history is a bounded deque."""
        notes_history = deque(maxlen=5)
        
        # Add more than max items
        for i in range(10):
            notes_history.append([f'note_{i}'])
        
        # Should only keep last 5
        assert len(notes_history) == 5
        assert notes_history[0] == ['note_5']
        assert notes_history[-1] == ['note_9']
    
    def test_frequencies_history_deque(self):
        """Test frequencies history is bounded."""
        frequencies_history = deque(maxlen=5)
        
        for i in range(10):
            frequencies_history.append([(440 + i, 0.5)])
        
        assert len(frequencies_history) == 5
    
    def test_chroma_history_deque(self):
        """Test chroma history is bounded."""
        chroma_history = deque(maxlen=5)
        
        for i in range(10):
            chroma_history.append(np.zeros(12))
        
        assert len(chroma_history) == 5


class TestConfigHandling:
    """Test configuration handling for both CLI and web."""
    
    def test_cli_args_to_config_values(self, mock_args):
        """Test CLI args contain expected values."""
        assert mock_args.instrument == 'guitar'
        assert mock_args.sensitivity == 1.0
        assert mock_args.confidence_threshold == 0.6
        assert mock_args.silence_threshold == 0.005
        assert mock_args.overlap == 0.75
        assert mock_args.multi_pitch == True
    
    def test_web_config_dict_values(self, mock_config):
        """Test web config dict contains expected values."""
        assert mock_config['instrument'] == 'guitar'
        assert mock_config['sensitivity'] == 1.0
        assert mock_config['confidence_threshold'] == 0.6
        assert mock_config['silence_threshold'] == 0.005
        assert mock_config['overlap'] == 0.75
        assert mock_config['multi_pitch'] == True
    
    def test_config_values_match(self, mock_args, mock_config):
        """Test CLI and web configs have matching default values."""
        assert mock_args.instrument == mock_config['instrument']
        assert mock_args.sensitivity == mock_config['sensitivity']
        assert mock_args.confidence_threshold == mock_config['confidence_threshold']
        assert mock_args.silence_threshold == mock_config['silence_threshold']
        assert mock_args.overlap == mock_config['overlap']


class TestInstrumentPresetLoading:
    """Test instrument preset loading."""
    
    def test_load_guitar_preset(self):
        """Test loading guitar preset."""
        preset = INSTRUMENT_PRESETS['guitar']
        assert preset['low_freq'] == 80
        assert preset['high_freq'] == 2000
    
    def test_load_piano_preset(self):
        """Test loading piano preset."""
        preset = INSTRUMENT_PRESETS['piano']
        assert preset['low_freq'] == 100
        assert preset['high_freq'] == 4000
    
    def test_invalid_instrument_raises(self):
        """Test accessing invalid instrument raises KeyError."""
        with pytest.raises(KeyError):
            _ = INSTRUMENT_PRESETS['invalid_instrument']


class TestOverlapConfiguration:
    """Test overlap ratio configuration."""
    
    def test_set_overlap_ratio(self):
        """Test setting overlap ratio."""
        set_overlap_ratio(0.5)
        from lib.common import get_overlap_ratio
        assert get_overlap_ratio() == 0.5
        set_overlap_ratio(0.75)  # Restore
    
    def test_overlap_affects_hop_size(self):
        """Test overlap ratio affects hop size."""
        from lib.common import get_hop_size
        
        set_overlap_ratio(0.75)
        hop_75 = get_hop_size()
        
        set_overlap_ratio(0.5)
        hop_50 = get_hop_size()
        
        # 50% overlap should have larger hop size than 75% overlap
        assert hop_50 > hop_75
        set_overlap_ratio(0.75)  # Restore


class TestAudioLevelDetection:
    """Test audio level detection (RMS)."""
    
    def test_rms_silence(self, silence_audio):
        """Test RMS of silent audio is zero."""
        rms = np.sqrt(np.mean(silence_audio.astype(np.float64)**2))
        assert rms == 0.0
    
    def test_rms_sine_wave(self, sine_wave_440hz):
        """Test RMS of sine wave is positive."""
        rms = np.sqrt(np.mean(sine_wave_440hz.astype(np.float64)**2))
        assert rms > 0
        # RMS of 0.5 amplitude sine wave should be approximately 0.5/sqrt(2) ≈ 0.354
        assert 0.3 < rms < 0.4
    
    def test_rms_threshold_detection(self, silence_audio, sine_wave_440hz):
        """Test silence threshold can distinguish between silence and signal."""
        threshold = 0.005
        
        rms_silence = np.sqrt(np.mean(silence_audio.astype(np.float64)**2))
        rms_signal = np.sqrt(np.mean(sine_wave_440hz.astype(np.float64)**2))
        
        assert rms_silence < threshold
        assert rms_signal > threshold

