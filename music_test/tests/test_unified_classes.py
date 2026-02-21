"""
Tests for the unified config, state, and output handler classes.
"""
import pytest
import numpy as np
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.config import AudioConfig
from lib.state import AudioProcessingState
from lib.output import ConsoleOutputHandler, DictOutputHandler
from lib.audio_processing import process_audio_chunk
from lib.common import get_chunk, get_buffer_size


class TestAudioConfig:
    """Test AudioConfig class."""
    
    def test_config_from_dict(self, mock_config):
        """Test creating config from dictionary."""
        config = AudioConfig(mock_config)
        
        assert config.instrument == 'guitar'
        assert config.sensitivity == 1.0
        assert config.confidence_threshold == 0.6
        assert config.silence_threshold == 0.005
        assert config.overlap == 0.75
    
    def test_config_from_args(self, mock_args):
        """Test creating config from argparse namespace."""
        config = AudioConfig(mock_args)
        
        assert config.instrument == 'guitar'
        assert config.sensitivity == 1.0
        assert config.confidence_threshold == 0.6
        assert config.silence_threshold == 0.005
        assert config.overlap == 0.75
    
    def test_config_get_method(self, mock_config):
        """Test config.get() method."""
        config = AudioConfig(mock_config)
        
        assert config.get('instrument') == 'guitar'
        assert config.get('nonexistent', 'default') == 'default'
    
    def test_config_instrument_preset_lookup(self):
        """Test instrument preset lookup for frequencies."""
        config = AudioConfig({'instrument': 'piano'})
        
        assert config.low_freq == 100  # Piano preset
        assert config.high_freq == 4000  # Piano preset
        assert config.instrument_name == 'Piano'
    
    def test_config_custom_frequencies_override(self):
        """Test custom frequencies override preset."""
        config = AudioConfig({
            'instrument': 'guitar',
            'low_freq': 150,
            'high_freq': 3000
        })
        
        assert config.low_freq == 150
        assert config.high_freq == 3000
    
    def test_config_to_dict(self, mock_config):
        """Test converting config to dictionary."""
        config = AudioConfig(mock_config)
        result = config.to_dict()
        
        assert isinstance(result, dict)
        assert result['instrument'] == 'guitar'
        assert result['sensitivity'] == 1.0


class TestAudioProcessingState:
    """Test AudioProcessingState class."""
    
    def test_state_initialization(self, mock_config):
        """Test state initialization."""
        state = AudioProcessingState(mock_config)
        
        assert len(state.audio_buffer) == get_buffer_size()
        assert state.buffer_index == 0
        assert len(state.notes_history) == 0
        assert state.last_chord is None
        assert state.chord_stability == 0
    
    def test_state_buffer_update(self, mock_config, chunk_size):
        """Test buffer update."""
        state = AudioProcessingState(mock_config)
        new_samples = np.ones(chunk_size, dtype=np.float32) * 0.5
        
        state.update_buffer(new_samples, chunk_size)
        
        assert state.buffer_index == chunk_size
        assert np.all(state.audio_buffer[:chunk_size] == 0.5)
    
    def test_state_add_detection(self, mock_config):
        """Test adding detection to history."""
        state = AudioProcessingState(mock_config)
        
        state.add_detection(['C', 'E', 'G'], [(261.63, 0.5), (329.63, 0.4)], np.zeros(12))
        
        assert len(state.notes_history) == 1
        assert state.notes_history[0] == ['C', 'E', 'G']
    
    def test_state_chord_stability(self, mock_config):
        """Test chord stability tracking."""
        state = AudioProcessingState(mock_config)
        
        stability = state.update_chord_stability('C')
        assert stability == 0
        assert state.last_chord == 'C'
        
        stability = state.update_chord_stability('C')
        assert stability == 1
        
        stability = state.update_chord_stability('G')
        assert stability == 0
        assert state.last_chord == 'G'
    
    def test_state_reset(self, mock_config, chunk_size):
        """Test state reset."""
        state = AudioProcessingState(mock_config)
        
        # Modify state
        state.update_buffer(np.ones(chunk_size, dtype=np.float32), chunk_size)
        state.add_detection(['C'], [(261.63, 0.5)], None)
        state.update_chord_stability('C')
        
        # Reset
        state.reset()
        
        assert state.buffer_index == 0
        assert len(state.notes_history) == 0
        assert state.last_chord is None
    
    def test_state_has_enough_history(self, mock_config):
        """Test history check."""
        state = AudioProcessingState(mock_config)
        
        assert not state.has_enough_history(3)
        
        for i in range(3):
            state.add_detection([f'note_{i}'], [], None)
        
        assert state.has_enough_history(3)


class TestDictOutputHandler:
    """Test DictOutputHandler class."""
    
    def test_chord_detected_output(self, mock_config):
        """Test chord detection output."""
        handler = DictOutputHandler()
        
        result = handler.chord_detected(
            chord_name='C',
            confidence=0.85,
            stability=2,
            detected_notes=['C', 'E', 'G'],
            detected_frequencies=[(261.63, 0.5), (329.63, 0.4), (392.0, 0.3)],
            chroma_vector=np.zeros(12),
            config=mock_config
        )
        
        assert result['type'] == 'chord'
        assert result['chord'] == 'C'
        assert result['confidence'] == 0.85
        assert result['stability'] == 2
        assert 'timestamp' in result
    
    def test_notes_detected_output(self, mock_config):
        """Test notes detection output."""
        handler = DictOutputHandler()
        
        result = handler.notes_detected(
            detected_notes=['A', 'C', 'E'],
            detected_frequencies=[(440, 0.5), (523.25, 0.4), (659.25, 0.3)],
            chroma_vector=None,
            config=mock_config
        )
        
        assert result['type'] == 'notes'
        assert 'notes' in result
        assert 'timestamp' in result
    
    def test_no_detection_output(self, mock_config):
        """Test no detection output."""
        handler = DictOutputHandler()
        
        result = handler.no_detection(mock_config)
        
        assert result is None
    
    def test_listening_output(self, mock_config):
        """Test listening status output."""
        handler = DictOutputHandler()
        
        result = handler.listening(mock_config)
        
        assert result['type'] == 'listening'
        assert 'timestamp' in result


class TestConsoleOutputHandler:
    """Test ConsoleOutputHandler class."""
    
    def test_handler_creation(self, mock_config):
        """Test handler can be created."""
        handler = ConsoleOutputHandler(mock_config)
        
        assert handler.log_mode == mock_config.get('log', False)
        assert handler.debug == mock_config.get('debug', False)
    
    def test_chord_detected_returns_none(self, mock_config):
        """Test chord detection returns None (prints to stdout)."""
        handler = ConsoleOutputHandler(mock_config)
        
        result = handler.chord_detected(
            chord_name='C',
            confidence=0.85,
            stability=2,
            detected_notes=['C', 'E', 'G'],
            detected_frequencies=[(261.63, 0.5)],
            chroma_vector=None,
            config=mock_config
        )
        
        # Console handler prints, doesn't return
        assert result is None


class TestProcessAudioChunk:
    """Test the core process_audio_chunk function."""
    
    def test_process_silence(self, silence_audio, mock_config):
        """Test processing silence."""
        state = AudioProcessingState(mock_config)
        
        result = process_audio_chunk(
            silence_audio, state, mock_config,
            output_handler=None, debug_log_level='INFO'
        )
        
        # Should return None for silence
        assert result is None
    
    def test_process_with_dict_handler(self, sine_wave_440hz, mock_config):
        """Test processing with DictOutputHandler."""
        state = AudioProcessingState(mock_config)
        handler = DictOutputHandler()
        
        # Process multiple chunks to build history
        for _ in range(5):
            result = process_audio_chunk(
                sine_wave_440hz, state, mock_config,
                output_handler=handler, debug_log_level='INFO'
            )
        
        # Should have processed without error
        # Result may be None if no chord detected or dict if chord detected
        assert result is None or isinstance(result, dict)
    
    def test_process_updates_state(self, sine_wave_440hz, mock_config):
        """Test that processing updates state."""
        state = AudioProcessingState(mock_config)
        initial_buffer_index = state.buffer_index
        
        process_audio_chunk(
            sine_wave_440hz, state, mock_config,
            output_handler=None, debug_log_level='INFO'
        )
        
        # Buffer index should have changed
        assert state.buffer_index != initial_buffer_index


class TestConfigConsistency:
    """Test that CLI and web configs produce same results."""
    
    def test_cli_web_config_equivalence(self, mock_args, mock_config):
        """Test CLI args and web config produce equivalent AudioConfig."""
        cli_config = AudioConfig(mock_args)
        web_config = AudioConfig(mock_config)
        
        assert cli_config.sensitivity == web_config.sensitivity
        assert cli_config.confidence_threshold == web_config.confidence_threshold
        assert cli_config.silence_threshold == web_config.silence_threshold
        assert cli_config.overlap == web_config.overlap
        assert cli_config.multi_pitch == web_config.multi_pitch
        assert cli_config.progression == web_config.progression
    
    def test_state_behavior_identical(self, mock_args, mock_config, chunk_size):
        """Test state behaves identically with CLI and web config."""
        cli_state = AudioProcessingState(mock_args)
        web_state = AudioProcessingState(mock_config)
        
        test_samples = np.random.randn(chunk_size).astype(np.float32)
        
        cli_state.update_buffer(test_samples, chunk_size)
        web_state.update_buffer(test_samples, chunk_size)
        
        assert cli_state.buffer_index == web_state.buffer_index
        np.testing.assert_array_equal(cli_state.audio_buffer, web_state.audio_buffer)

